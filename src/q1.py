import argparse, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from common import *
from models import MLP

def run(args):
    seed_everything(); device=get_device(args.device)
    tr,va,te,*_=prepare_data(args.data_dir,args.batch_size,args.num_workers)
    out=Path(args.output_dir)/"q1"; out.mkdir(parents=True,exist_ok=True)
    # Q1(a): test capacity/regularisation as competing explanations.
    # Similar-width models are reported with exact parameter counts.
    configs={"mlp_256":MLP((256,128),0.0),
             "mlp_256_dropout":MLP((256,128),0.35)}
    histories={}
    for name,model in configs.items():
        print(name,"parameters",count_parameters(model))
        p,h=fit_classifier(model,tr,va,args.epochs,args.lr,args.weight_decay,
                           device,out,name)
        histories[name]=h
        model=load_best(model,p,device)
        test=evaluate(model,te,nn.CrossEntropyLoss(),device)
        save_json({"test":test,"parameters":count_parameters(model)},out/f"{name}_test.json")
    plot_histories(histories,out/"q1a_curves.png","Q1(a): MLP capacity/regularisation comparison")

    # Q1(b): random-label memorisation. Keep validation labels true.
    # Use fixed random targets for the 45k training subset.
    train_subset=tr.dataset
    xs=[]; ys=[]
    for x,y in DataLoader(train_subset,batch_size=512,shuffle=False):
        xs.append(x); ys.append(y)
    X=torch.cat(xs); Y=torch.cat(ys).view(-1)
    g=torch.Generator().manual_seed(SEED)
    random_y=torch.randint(0,10,(len(Y),),generator=g)
    random_loader=DataLoader(TensorDataset(X,random_y),batch_size=args.batch_size,shuffle=True)
    model=MLP((2048,1024),0.0).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0)
    criterion=nn.CrossEntropyLoss(); rows=[]
    for ep in range(1,args.mem_epochs+1):
        trm=train_epoch(model,random_loader,opt,criterion,device)
        val=evaluate(model,va,criterion,device)
        rows.append({"epoch":ep,"random_train_accuracy":trm["accuracy"],
                     "true_label_val_accuracy":val["accuracy"],
                     "random_train_loss":trm["loss"],"true_label_val_loss":val["loss"]})
        print("memorisation",rows[-1])
    import pandas as pd, matplotlib.pyplot as plt
    pd.DataFrame(rows).to_csv(out/"q1b_memorisation.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([r["epoch"] for r in rows],[r["random_train_accuracy"] for r in rows],label="random-label train")
    ax.plot([r["epoch"] for r in rows],[r["true_label_val_accuracy"] for r in rows],label="true-label validation")
    ax.set(xlabel="Epoch",ylabel="Accuracy",title="Q1(b): Memorisation trajectory"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/"q1b_memorisation.png",dpi=180); plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="./data"); p.add_argument("--output-dir",default="./outputs")
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--mem-epochs",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=128); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--device",default="auto"); run(p.parse_args())
