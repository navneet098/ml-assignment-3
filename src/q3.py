import argparse, torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from common import *
from models import ConvAutoencoder, CNN

def train_ae(model,loader,val_loader,epochs,device,out,name,denoise=False,sigma=.15):
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    criterion=nn.MSELoss(); best=float("inf"); history=[]
    for ep in range(1,epochs+1):
        model.train(); s=n=0
        for x,_ in loader:
            x=x.to(device); inp=(x+sigma*torch.randn_like(x)).clamp(0,1) if denoise else x
            opt.zero_grad(set_to_none=True); loss=criterion(model(inp),x); loss.backward(); opt.step()
            s+=loss.item()*len(x); n+=len(x)
        model.eval(); vs=vn=0
        with torch.no_grad():
            for x,_ in val_loader:
                x=x.to(device); inp=(x+sigma*torch.randn_like(x)).clamp(0,1) if denoise else x
                vs+=criterion(model(inp),x).item()*len(x); vn+=len(x)
        row={"epoch":ep,"train_mse":s/n,"val_mse":vs/vn}; history.append(row); print(name,row)
        if vs/vn<best:
            best=vs/vn; torch.save(model.state_dict(),out/f"{name}_best.pt")
    import pandas as pd
    pd.DataFrame(history).to_csv(out/f"{name}_history.csv",index=False)
    return best

@torch.no_grad()
def classify_reconstructions(ae,classifier,loader,device,noise=False,sigma=.15):
    ae.eval(); classifier.eval(); correct=n=0
    for x,y in loader:
        x=x.to(device); y=y.view(-1).to(device)
        inp=(x+sigma*torch.randn_like(x)).clamp(0,1) if noise else x
        recon=ae(inp)
        pred=classifier(recon).argmax(1)
        correct+=(pred==y).sum().item(); n+=len(y)
    return correct/n

def run(args):
    seed_everything(); device=get_device(args.device)
    tr,va,te,*_=prepare_data(args.data_dir,args.batch_size,args.num_workers)
    out=Path(args.output_dir)/"q3"; out.mkdir(parents=True,exist_ok=True)
    dims=[64,512,4096]; val_errors={}
    for d in dims:
        ae=ConvAutoencoder(d)
        err=train_ae(ae,tr,va,args.epochs,device,out,f"ae_{d}")
        val_errors[str(d)]=err
    best_dim=int(min(val_errors,key=val_errors.get))
    best_ae=ConvAutoencoder(best_dim).to(device)
    best_ae.load_state_dict(torch.load(out/f"ae_{best_dim}_best.pt",map_location=device))
    # Test reconstruction MSE.
    best_ae.eval(); mse=nn.MSELoss(reduction="sum"); s=n=0
    with torch.no_grad():
        for x,_ in te:
            x=x.to(device); s+=mse(best_ae(x),x).item(); n+=x.numel()
    test_mse=s/n
    # Denoising AE; same architecture as selected clean AE.
    dae=ConvAutoencoder(best_dim)
    train_ae(dae,tr,va,args.epochs,device,out,"denoising_ae",denoise=True)
    dae.load_state_dict(torch.load(out/"denoising_ae_best.pt",map_location=device))
    # Fixed CNN classifier trained on clean images. Train here if Q2 checkpoint absent.
    cnn=CNN((128,256,512),512).to(device); ckpt=Path(args.output_dir)/"q2"/"CNN_clean_best.pt"
    if not ckpt.exists():
        from common import fit_classifier
        ckpt,_=fit_classifier(cnn,tr,va,args.classifier_epochs,1e-3,1e-4,device,
                              Path(args.output_dir)/"q2","CNN_clean")
    cnn.load_state_dict(torch.load(ckpt,map_location=device)["model"])
    dae_acc=classify_reconstructions(dae,cnn,te,device,noise=True)
    clean_ae_acc=classify_reconstructions(best_ae,cnn,te,device)
    noisy_correct=noisy_n=0; cnn.eval()
    with torch.no_grad():
        for x,y in te:
            x=x.to(device); y=y.view(-1).to(device)
            xn=(x+.15*torch.randn_like(x)).clamp(0,1)
            noisy_correct+=(cnn(xn).argmax(1)==y).sum().item(); noisy_n+=len(y)
    # Linear probe on frozen representations, no encoder fine tuning.
    feats=[]; labels=[]
    best_ae.eval()
    with torch.no_grad():
        for x,y in tr:
            feats.append(best_ae.encode(x.to(device)).cpu()); labels.append(y.view(-1))
    Xtr=torch.cat(feats).numpy(); ytr=torch.cat(labels).numpy()
    feats=[]; labels=[]
    with torch.no_grad():
        for x,y in te:
            feats.append(best_ae.encode(x.to(device)).cpu()); labels.append(y.view(-1))
    Xte=torch.cat(feats).numpy(); yte=torch.cat(labels).numpy()
    from sklearn.linear_model import LogisticRegression
    probe=LogisticRegression(max_iter=1000,solver="lbfgs")
    probe.fit(Xtr,ytr); probe_acc=probe.score(Xte,yte)
    save_json({"validation_mse_by_bottleneck":val_errors,"selected_bottleneck":best_dim,
               "selected_clean_ae_test_mse":test_mse,
               "cnn_accuracy_noisy":noisy_correct/noisy_n,
               "cnn_accuracy_clean_ae_reconstruction":clean_ae_acc,
               "cnn_accuracy_denoising_ae_reconstruction":dae_acc,
               "frozen_encoder_linear_probe_test_accuracy":probe_acc},out/"q3_results.json")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="./data"); p.add_argument("--output-dir",default="./outputs")
    p.add_argument("--epochs",type=int,default=40); p.add_argument("--classifier-epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=128); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--device",default="auto"); run(p.parse_args())
