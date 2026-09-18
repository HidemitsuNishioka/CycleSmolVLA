"""HAMLET modality registration for the SO-101 single-camera dataset."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so101_joint = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["top"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
    ),
    "action": ModalityConfig(
        # GR00T-N1.6's default action head horizon is 16. The raw v3 dataset
        # has longer episodes, so the converter keeps all frames while this
        # modality selects 16-step training chunks.
        delta_indices=list(range(16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="arm",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(so101_joint, EmbodimentTag.NEW_EMBODIMENT)
