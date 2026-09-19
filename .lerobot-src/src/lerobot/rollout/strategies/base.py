# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import sys
import time
from threading import Event

from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.keyboard_input import TerminalKeyListener

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.
    """

    def __init__(self, config):
        super().__init__(config)
        self._restart_requested = Event()

    def _on_key(self, key: str) -> None:
        # The keyboard thread only requests a restart; the control thread owns it.
        if key.lower() == "r":
            self._restart_requested.set()

    def _restart_inference(self, ctx: RolloutContext, timer: CycleTimer) -> bool:
        logger.info("Restart requested: waiting for the current inference to finish")
        self._engine.pause()
        # pause() alone does not wait for an in-flight RTC chunk. Join the worker
        # before touching policy/processors, so an old result cannot enter the new queue.
        self._engine.stop()
        if ctx.runtime.shutdown_event.is_set():
            return False
        self.reset_control_state()
        self._engine.start()
        self._engine.resume()
        timer.restart()
        logger.info("Inference restarted: observation history and actions cleared; duration timer restarted")
        return True

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine."""
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        timer = CycleTimer(
            cfg.fps,
            interpolator.multiplier,
            records_data=False,
            report=ctx.runtime.cadence_report,
        )

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        listener = None
        try:
            if self.config.keyboard_restart and not getattr(cfg, "interactive", False):
                if sys.stdin.isatty():
                    listener = TerminalKeyListener(self._on_key)
                    listener.start()
                    logger.info("Keyboard: focus this terminal and press R to reset history/actions and restart")
                else:
                    logger.warning("Keyboard restart unavailable: stdin is not a terminal")
            while not ctx.runtime.shutdown_event.is_set():
                if self._restart_requested.is_set():
                    self._restart_requested.clear()
                    if not self._restart_inference(ctx, timer):
                        break
                    start_time = time.perf_counter()
                timer.tick(new_cycle=interpolator.needs_new_action())

                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                with timer.section("observe"):
                    obs = robot.get_observation()
                with timer.section("process_obs"):
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                if self._handle_warmup(cfg.use_torch_compile, timer):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)
                with timer.section("telemetry"):
                    self._log_telemetry(obs_processed, action_dict, ctx.runtime)

                # Service the text-query channel (/vqa answers, /autosteer turns) at
                # the end of the tick; no-op when nothing is queued.
                with timer.section("query"):
                    engine.pump_query(obs_processed)

                timer.wait()
        finally:
            if listener is not None:
                listener.stop()
            logger.info("Base strategy control loop ended")
            timer.log_run_summary()

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        self._teardown_hardware(
            ctx.hardware,
            return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
        )
        logger.info("Base strategy teardown complete")
