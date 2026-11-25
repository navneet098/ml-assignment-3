import argparse, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from common import *
from models import MLP,CNN

class PermutedCIFAR(Dataset):
    """Same fixed permutation for all images; RGB triplets move together."""
    def __init__(self, base, permutation):
        self.base=base; self.permutation=torch.as_tensor(permutation,dtype=torch.long)
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]
        x=x.reshape(3,1024)[:,self.permutation].reshape(3,32,32)
        return x,y

def run(args):
    seed_everything(); device=get_device(args.device)
    tr,va,te,train_ds,test_ds,train_idx,val_idx=prepare_data(
        args.data_dir,args.batch_size,args.num_workers)
    out=Path(args.output_dir)/"q2"; out.mkdir(parents=True,exist_ok=True)
    # Shared fixed permutation for train, validation and test.
    perm=np.random.default_rng(SEED+2).permutation(1024)
    np.save(out/"spatial_permutation.npy",perm)
    # Validation/test subsets correspond to exact original indices.
    # train_ds is the torchvision full train dataset with ToTensor transform.
    from torch.utils.data import Subset
    trp=DataLoader(PermutedCIFAR(Subset(train_ds,train_idx),perm),batch_size=args.batch_size,shuffle=True)
    vap=DataLoader(PermutedCIFAR(Subset(train_ds,val_idx),perm),batch_size=args.batch_size,shuffle=False)
    tep=DataLoader(PermutedCIFAR(test_ds,perm),batch_size=args.batch_size,shuffle=False)
    # Clean and disturbed conditions, same architectures and training protocol.
    model_specs={"MLP":MLP((512,256)),"CNN":CNN((128,256,512),512)}
    # Clean condition first.
    clean_results={}
    for name,model in model_specs.items():
        print(name,"parameters",count_parameters(model))
        p,h=fit_classifier(model,tr,va,args.epochs,args.lr,args.weight_decay,device,out,name+"_clean")
        model=load_best(model,p,device)
        clean_results[name]=evaluate(model,te,nn.CrossEntropyLoss(),device,collect=True)
        save_json({"parameters":count_parameters(model),
                   "test_accuracy":clean_results[name]["accuracy"]},out/f"{name}_clean_test.json")
    # Retrain both models on the spatially permuted images.
    disturbed_results={}
    for name,model in {"MLP":MLP((512,256)),"CNN":CNN((128,256,512),512)}.items():
        p,h=fit_classifier(model,trp,vap,args.epochs,args.lr,args.weight_decay,device,out,name+"_disturbed")
        model=load_best(model,p,device)
        disturbed_results[name]=evaluate(model,tep,nn.CrossEntropyLoss(),device,collect=True)
        save_json({"parameters":count_parameters(model),
                   "test_accuracy":disturbed_results[name]["accuracy"]},out/f"{name}_disturbed_test.json")
    # Q2(c): local corruption/occlusion experiment, same fixed CNN classifier.
    # A fixed 12x12 central square is zeroed at test time; analyse confidence.
    cnn=load_best(CNN((128,256,512),512),out/"CNN_clean_best.pt",device)
    cnn.eval(); ys=[]; preds=[]; confs=[]
    for x,y in te:
        x=x.to(device); x[:,:,10:22,10:22]=0.0
        with torch.no_grad():
            prob=cnn(x).softmax(1)
        ys.extend(y.view(-1).tolist()); preds.extend(prob.argmax(1).cpu().tolist())
        confs.extend(prob.max(1).values.cpu().tolist())
    from sklearn.metrics import accuracy_score, confusion_matrix
    occ_acc=accuracy_score(ys,preds)
    save_json({"condition":"central 12x12 zero occlusion","test_accuracy":occ_acc,
               "mean_confidence":float(np.mean(confs)),
               "correct_mean_confidence":float(np.mean([c for c,p,y in zip(confs,preds,ys) if p==y])),
               "incorrect_mean_confidence":float(np.mean([c for c,p,y in zip(confs,preds,ys) if p!=y])),
               "confusion_matrix":confusion_matrix(ys,preds).tolist()},out/"q2c_occlusion.json")
    # Compact summary
    summary=[]
    for name in ["MLP","CNN"]:
        summary.append({"model":name,"clean":clean_results[name]["accuracy"],
                        "permuted":disturbed_results[name]["accuracy"],
                        "drop":clean_results[name]["accuracy"]-disturbed_results[name]["accuracy"]})
    import pandas as pd
    pd.DataFrame(summary).to_csv(out/"q2_summary.csv",index=False)
    print("Q2 summary",summary,"occlusion accuracy",occ_acc)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="./data"); p.add_argument("--output-dir",default="./outputs")
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--num-workers",type=int,default=2); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--device",default="auto")
    run(p.parse_args())
