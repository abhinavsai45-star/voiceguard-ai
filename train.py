"""Train and evaluate the included CNN from labelled audio folders.

Expected layout: dataset/real/*.wav and dataset/fake/*.wav (MP3/FLAC accepted).
The script makes a stratified train/validation split and reports held-out metrics.
"""
import argparse, random, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,classification_report
from ml import AudioCNN,features_from_bytes

EXTS={'.wav','.mp3','.flac','.ogg','.m4a'}
class AudioSet(Dataset):
    def __init__(self,items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]
        try: x,_=features_from_bytes(p.read_bytes(),p.name)
        except Exception as e: raise RuntimeError(f'Could not load {p}: {e}') from e
        return x.squeeze(0),torch.tensor(y,dtype=torch.long)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='dataset');ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--seed',type=int,default=7);a=ap.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    root=Path(a.data); items=[]
    for label,folder in enumerate(('real','fake')): items += [(p,label) for p in (root/folder).rglob('*') if p.suffix.lower() in EXTS]
    counts=[sum(y==i for _,y in items) for i in (0,1)]
    if min(counts)<10: raise SystemExit(f'Need at least 10 files in each folder; found real={counts[0]}, fake={counts[1]}. Use a labelled dataset with speaker-disjoint splits where possible.')
    # Cache features once; repeatedly decoding audio for every epoch is wasteful.
    features=[]
    for i,(p,y) in enumerate(items,1):
        try: x,_=features_from_bytes(p.read_bytes(),p.name)
        except Exception as e: raise SystemExit(f'Could not process {p}: {e}') from e
        features.append((x.squeeze(0),y,p.stem))
        if i%100==0: print(f'Prepared {i}/{len(items)} audio files')
    groups=np.array([g for _,_,g in features]);indices=np.arange(len(features))
    train_idx,val_idx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=a.seed).split(indices,groups=groups))
    train=[(features[i][0],features[i][1]) for i in train_idx];val=[(features[i][0],features[i][1]) for i in val_idx]
    if {y for _,y in train}!={0,1} or {y for _,y in val}!={0,1}: raise SystemExit('The grouped split did not include both labels; add more varied files and retry.')
    print(f'Grouped split: train={len(train)}, validation={len(val)}; duplicate filenames stay together.')
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=AudioCNN().to(device)
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0)
    vloader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);loss_fn=nn.CrossEntropyLoss()
    best=0.;best_truth=[];best_pred=[];out=Path('models/voiceguard_cnn.pt');out.parent.mkdir(exist_ok=True)
    for epoch in range(a.epochs):
        model.train();running=0
        for x,y in loader:
            x,y=x.to(device),y.to(device);optimizer.zero_grad();loss=loss_fn(model(x),y);loss.backward();optimizer.step();running+=loss.item()
        model.eval();truth=[];pred=[]
        with torch.no_grad():
            for x,y in vloader: pred.extend(model(x.to(device)).argmax(1).cpu().tolist());truth.extend(y.tolist())
        acc=accuracy_score(truth,pred);print(f'Epoch {epoch+1}/{a.epochs} loss={running/max(1,len(loader)):.4f} validation_accuracy={acc:.3f}')
        if acc>=best:
            best=acc;best_truth=truth[:];best_pred=pred[:];torch.save({'model':model.cpu().state_dict(),'validation_accuracy':best,'classes':['REAL VOICE','AI-GENERATED VOICE'],'sample_rate':16000},out);model.to(device)
    print('\nHeld-out validation report (not a guarantee of real-world performance):')
    report=classification_report(best_truth,best_pred,target_names=['REAL VOICE','AI-GENERATED VOICE'],zero_division=0,output_dict=True)
    print(classification_report(best_truth,best_pred,target_names=['REAL VOICE','AI-GENERATED VOICE'],zero_division=0))
    metrics={'validation_accuracy':best,'validation_samples':len(best_truth),'class_report':report,
             'classes':['REAL VOICE','AI-GENERATED VOICE'],'dataset':'MLAAD-tiny education subset'}
    (out.parent/'voiceguard_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(f'Best validation accuracy: {best:.3f}; saved {out}')
if __name__=='__main__': main()
