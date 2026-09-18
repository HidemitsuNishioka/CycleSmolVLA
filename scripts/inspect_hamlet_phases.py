"""Render recorded trajectories and contact sheets without any robot connection."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('HAMLET-Isaac-GR00T/data/so101_shake_cup_gr00t')
OUT = Path('outputs/eval/hamlet_phase_review')
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(10, 1, figsize=(16, 23), sharex=True)
    for ep, ax in enumerate(axes):
        df = pd.read_parquet(ROOT / f'data/chunk-000/episode_{ep:06d}.parquet')
        actions = np.stack(df.actions)
        for j, name in enumerate(NAMES):
            ax.plot(np.arange(len(df))/30, actions[:, j], label=name, linewidth=1)
        ax.set_title(f'Episode {ep} - recorded actions', fontsize=10)
        ax.set_xticks(np.arange(0, 15.1, .5))
        ax.grid(alpha=.3)
        cap = cv2.VideoCapture(str(ROOT / f'videos/chunk-000/top/episode_{ep:06d}.mp4'))
        tiles = []
        for frame in range(0, 450, 15):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError((ep, frame))
            img = cv2.resize(img, (256, 192))
            cv2.putText(img, f'ep{ep} f{frame} {frame/30:.1f}s', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            tiles.append(img)
        cap.release()
        cv2.imwrite(str(OUT / f'episode_{ep:03d}_contact.jpg'), np.vstack([np.hstack(tiles[i:i+6]) for i in range(0, 30, 6)]))
    axes[0].legend(ncol=6, fontsize=8)
    axes[-1].set_xlabel('Recorded time [s]')
    fig.tight_layout()
    fig.savefig(OUT / 'recorded_actions.png', dpi=120)
    plt.close(fig)

if __name__ == '__main__':
    main()
