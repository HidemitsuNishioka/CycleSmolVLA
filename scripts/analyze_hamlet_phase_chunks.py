"""Analyze saved teacher-observation predictions; never connects to hardware.

Phase boundaries are diagnostic annotations, not dataset-provided labels.
Each prediction is assigned by its target frame (anchor + horizon index).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import find_peaks

ROOT = Path('HAMLET-Isaac-GR00T/data/so101_shake_cup_gr00t')
OUT = Path('outputs/eval/hamlet_phase_review')
WINDOWS = [(4.9,7.8),(4.7,7.8),(4.3,8.3),(4.3,7.9),(4.4,7.6),
           (3.9,7.2),(3.2,6.7),(4.,7.4),(3.4,6.7),(3.8,6.8)]
PHASES = ['preparation','shake_1','shake_2','shake_3','return','stopped']
COLORS = ['#eeeeee','#b6dfef','#c5e7b1','#ffd69c','#e6d7ed','#d1d1d1']

def metrics(pred, gt):
    error = pred-gt
    return dict(samples=len(error), arm_mae=float(np.abs(error[:,:5]).mean()),
                gripper_mae=float(np.abs(error[:,5]).mean()),
                mae_by_joint=np.abs(error).mean(axis=0).tolist(),
                arm_rmse=float(np.sqrt(np.square(error[:,:5]).mean())))

def main():
    groups = {name: [[],[],[]] for name in PHASES}
    annotations = []
    details = []
    predictions, targets, states = [],[],[]
    pure_stop = [[],[],[]]
    for ep,(lo,hi) in enumerate(WINDOWS):
        df = pd.read_parquet(ROOT / f'data/chunk-000/episode_{ep:06d}.parquet')
        a,s = np.stack(df.actions),np.stack(df.state)
        p,_ = find_peaks(a[:,0],prominence=8,distance=15)
        p = p[(p>=lo*30)&(p<=hi*30)]
        assert len(p)==3, (ep,p)
        # Approximate first onset at 0.6 s before first peak; later boundaries
        # are intervening troughs. Last trough search is limited to 0.8 s.
        bounds = [int(p[0])-18]+[int(x+np.argmin(a[x:y+1,0])) for x,y in zip(p[:-1],p[1:])]
        bounds += [int(p[-1]+np.argmin(a[p[-1]:p[-1]+25,0]))]
        bad = (np.abs(a-np.median(a[-30:],axis=0)).max(axis=1)>1) | (np.abs(s-np.median(s[-30:],axis=0)).max(axis=1)>1)
        stop = int(np.where(bad)[0][-1]+1)
        edges = [0]+bounds+[stop,len(a)]
        assert all(x<y for x,y in zip(edges[:-1],edges[1:])), edges
        labels = np.empty(len(a),dtype=int)
        for i,(x,y) in enumerate(zip(edges[:-1],edges[1:])):
            labels[x:y]=i
        annotations.append(dict(episode=ep, peaks=p.tolist(), boundaries=edges,
                                boundaries_seconds=(np.asarray(edges)/30).tolist()))
        z=np.load(OUT / f'episode_{ep:03d}_chunks.npz')
        pred,gt=z['predictions'],z['targets']
        assert pred.shape==gt.shape==(28,16,6)
        assert np.isfinite(pred).all()
        times=z['anchors'][:,None]+np.arange(16)[None,:]
        np.testing.assert_allclose(gt,a[times],rtol=0,atol=1e-5)
        np.testing.assert_allclose(z['states'],s[z['anchors']],rtol=0,atol=1e-5)
        hold=np.broadcast_to(z['states'][:,None,:],pred.shape)
        predictions.append(pred); targets.append(gt); states.append(hold)
        for i,name in enumerate(PHASES):
            mask=labels[times]==i
            for dst,src in zip(groups[name],(pred,gt,hold)):
                dst.append(src[mask])
            details.append(dict(episode=ep,phase=name,model=metrics(pred[mask],gt[mask]),hold=metrics(hold[mask],gt[mask])))
        stop_chunks=z['anchors']>=stop
        for dst,src in zip(pure_stop,(pred,gt,hold)):
            dst.append(src[stop_chunks].reshape(-1,6))
        fig,axes=plt.subplots(3,2,figsize=(16,10),sharex=True)
        for j,ax in enumerate(axes.flat):
            for i,(x,y) in enumerate(zip(edges[:-1],edges[1:])):
                ax.axvspan(x/30,y/30,color=COLORS[i],alpha=.5)
            ax.plot(np.arange(len(a))/30,a[:,j],color='black',lw=1.5,label='Recorded action')
            ax.plot(np.arange(len(s))/30,s[:,j],color='gray',lw=.8,label='Recorded state')
            for k,start in enumerate(z['anchors']):
                ax.plot((start+np.arange(16))/30,pred[k,:,j],color='#d14836',lw=.8,alpha=.8,label='Predicted 16-step chunk' if k==0 else None)
            ax.set_title(str(z['joint_names'][j])); ax.grid(alpha=.2)
            ax.set_ylabel('Dataset joint units'); ax.set_xlabel('Recorded time [s]')
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'Episode {ep}: all 16 predictions per chunk, recorded observations/history (not a rollout)')
        fig.legend(handles=[Patch(facecolor=c,label=p,alpha=.5) for p,c in zip(PHASES,COLORS)],loc='upper center',bbox_to_anchor=(.5,.965),ncol=6,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.935)); fig.savefig(OUT/f'episode_{ep:03d}_full_chunks.png',dpi=120); plt.close(fig)
    result={}
    for name,arrays in groups.items():
        pred,gt,hold=map(np.concatenate,arrays)
        result[name]=dict(model=metrics(pred,gt),hold=metrics(hold,gt))
    pred,gt,hold=map(np.concatenate,(predictions,targets,states))
    horizon=[dict(index=h,model=metrics(pred[:,h],gt[:,h]),hold=metrics(hold[:,h],gt[:,h])) for h in range(16)]
    sp,sg,sh=map(np.concatenate,pure_stop)
    report=dict(episodes=10,anchors=280,predicted_frames=4480,seed=6,
                train_membership='Dataset split declares train 0:10; checkpoint training manifest unavailable.',
                phase_method='Approximate shoulder-pan cycles: reviewed search windows, peaks prominence>=8, trough boundaries; first onset=peak-18 frames. Terminal stop: all state/action joints remain within 1 dataset unit of their own final-30-frame median. Phase assignment uses target frame, not anchor.',
                caveats=['Recorded observations with rolling memory, not a closed-loop success test.',
                         'Cycle boundaries approximate; terminal stop is after return, not immediately after shake 3.',
                         '448/450 frames per episode evaluated; last 2 omitted to keep full 16-step chunks.',
                         'One seed, no stochastic confidence interval. Arm and gripper reported separately.'],
                phases=result,by_horizon=horizon,annotations=annotations,by_episode=details,
                pure_stopped_chunks=dict(model=metrics(sp,sg),hold=metrics(sh,sg)))
    (OUT/'phase_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,key,title in zip(axes,['arm_mae','gripper_mae'],['Arm MAE (5 joints)','Gripper MAE']):
        ax.plot(range(16),[v['model'][key] for v in horizon],label='HAMLET',marker='o')
        ax.plot(range(16),[v['hold'][key] for v in horizon],label='Hold current',marker='.')
        ax.set_title(title); ax.set_xlabel('Prediction index (0 = first)'); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout();fig.savefig(OUT/'horizon_errors.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(phases=result,pure_stopped=report['pure_stopped_chunks'],by_horizon=horizon),indent=2))

if __name__=='__main__':
    main()
