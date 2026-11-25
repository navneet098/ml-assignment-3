from __future__ import annotations
import os, random, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split

ROLL_NUMBER = "m26csa004"
SEED = int("".join(ch for ch in ROLL_NUMBER if ch.isdigit()))
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(requested="auto"):
    if requested != "auto": return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(data_dir="./data", batch_size=128, num_workers=2, seed=SEED):
    """Download/load CIFAR10; persist the exact stratified 45k/5k split."""
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    train_ds = CIFAR10(root=str(data_dir), train=True, download=True)
    test_ds = CIFAR10(root=str(data_dir), train=False, download=True)
    targets = np.asarray(train_ds.targets)
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=5000, random_state=seed, stratify=targets
    )
    split_file = data_dir / f"split_seed_{seed}.npz"
    if split_file.exists():
        saved = np.load(split_file)
        train_idx, val_idx = saved["train_idx"], saved["val_idx"]
    else:
        np.savez(split_file, train_idx=train_idx, val_idx=val_idx)

    # Tensor conversion only; no augmentation, so every comparison sees same pixels.
    tfm = transforms.ToTensor()
    train_ds.transform = tfm; test_ds.transform = tfm
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers,
                             pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader, train_ds, test_ds, train_idx, val_idx

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=str)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, loader, optimizer, criterion, device, label_override=None):
    model.train(); total=correct=n=loss_sum=0
    for x,y in loader:
        x=x.to(device); y=y.view(-1).to(device)
        if label_override is not None:
            # Dataset indices are not exposed by loader; this path is intentionally
            # not used. Random-label memorisation uses a dedicated TensorDataset.
            raise RuntimeError("Use the dedicated random-label loader for memorisation.")
        optimizer.zero_grad(set_to_none=True)
        logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
        bs=y.size(0); n+=bs; loss_sum+=loss.item()*bs
        correct+=(logits.argmax(1)==y).sum().item()
    return {"loss":loss_sum/n, "accuracy":correct/n}

@torch.no_grad()
def evaluate(model, loader, criterion, device, collect=False):
    model.eval(); total=correct=loss_sum=0; ys=[]; preds=[]; confs=[]
    for x,y in loader:
        x=x.to(device); y=y.view(-1).to(device)
        logits=model(x); loss=criterion(logits,y)
        probs=logits.softmax(1); pred=probs.argmax(1)
        bs=y.size(0); total+=bs; loss_sum+=loss.item()*bs
        correct+=(pred==y).sum().item()
        if collect:
            ys.extend(y.cpu().tolist()); preds.extend(pred.cpu().tolist())
            confs.extend(probs.max(1).values.cpu().tolist())
    result={"loss":loss_sum/total, "accuracy":correct/total}
    if collect: result.update(y_true=ys, y_pred=preds, confidence=confs)
    return result

def fit_classifier(model, train_loader, val_loader, epochs, lr, weight_decay,
                   device, out_dir, name):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model.to(device); opt=torch.optim.AdamW(model.parameters(), lr=lr,
                                             weight_decay=weight_decay)
    criterion=torch.nn.CrossEntropyLoss()
    history=[]; best=-1; best_path=out/f"{name}_best.pt"
    for epoch in range(1, epochs+1):
        tr=train_epoch(model, train_loader, opt, criterion, device)
        va=evaluate(model, val_loader, criterion, device)
        row={"epoch":epoch, **{f"train_{k}":v for k,v in tr.items()},
             **{f"val_{k}":v for k,v in va.items()}}
        history.append(row); print(name, row)
        if va["accuracy"]>best:
            best=va["accuracy"]
            torch.save({"model":model.state_dict(),"epoch":epoch,
                        "val_accuracy":best}, best_path)
    import pandas as pd
    pd.DataFrame(history).to_csv(out/f"{name}_history.csv", index=False)
    return best_path, history

def load_best(model, path, device):
    ckpt=torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); return model

def plot_histories(histories, out_path, title):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for name,h in histories.items():
        ep=[r["epoch"] for r in h]
        axs[0].plot(ep,[r["train_accuracy"] for r in h],label=name+" train")
        axs[0].plot(ep,[r["val_accuracy"] for r in h],linestyle="--",label=name+" val")
        axs[1].plot(ep,[r["train_loss"] for r in h],label=name+" train")
        axs[1].plot(ep,[r["val_loss"] for r in h],linestyle="--",label=name+" val")
    axs[0].set_title("Accuracy"); axs[1].set_title("Cross-entropy loss")
    for ax in axs: ax.set_xlabel("Epoch"); ax.grid(alpha=.25); ax.legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_path,dpi=180); plt.close(fig)
