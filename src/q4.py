import argparse, numpy as np, torch, pandas as pd
from pathlib import Path
from common import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

def collect_flat(loader,device):
    xs=[]; ys=[]
    for x,y in loader:
        xs.append(x.view(len(x),-1).numpy()); ys.append(y.view(-1).numpy())
    return np.concatenate(xs),np.concatenate(ys)

def run(args):
    seed_everything()
    tr,va,te,*_=prepare_data(args.data_dir,args.batch_size,args.num_workers)
    out=Path(args.output_dir)/"q4"; out.mkdir(parents=True,exist_ok=True)
    Xtr,ytr=collect_flat(tr,"cpu"); Xv,yv=collect_flat(va,"cpu"); Xte,yte=collect_flat(te,"cpu")
    # Fit preprocessing on training data only.
    scaler=StandardScaler()
    Xtr=scaler.fit_transform(Xtr); Xv=scaler.transform(Xv); Xte=scaler.transform(Xte)
    pca=PCA(n_components=args.pca_components,random_state=SEED)
    Ztr=pca.fit_transform(Xtr); Zv=pca.transform(Xv); Zte=pca.transform(Xte)
    np.save(out/"pca_components.npy",pca.components_)
    # Fixed stratified SVM subset selected before hyperparameter comparison.
    rng=np.random.default_rng(SEED+4); chosen=[]
    for c in range(10):
        ids=np.where(ytr==c)[0]; chosen.extend(rng.choice(ids,min(args.subset_per_class,len(ids)),replace=False))
    chosen=np.array(sorted(chosen))
    np.save(out/"svm_train_subset_indices.npy",chosen)
    kernels=["linear","poly","rbf"]; Cs=[.01,.1,1,10,100]; rows=[]; models={}
    for kernel in kernels:
        for C in Cs:
            clf=SVC(C=C,kernel=kernel,gamma="scale",degree=3,cache_size=3000)
            clf.fit(Ztr[chosen],ytr[chosen])
            tr_acc=accuracy_score(ytr[chosen],clf.predict(Ztr[chosen]))
            val_acc=accuracy_score(yv,clf.predict(Zv))
            frac=len(clf.support_)/len(chosen)
            rows.append({"kernel":kernel,"C":C,"train_accuracy":tr_acc,
                         "validation_accuracy":val_acc,"support_vector_fraction":frac})
            models[(kernel,C)]=clf
            print(rows[-1])
    df=pd.DataFrame(rows); df.to_csv(out/"svm_sweep.csv",index=False)
    best=df.sort_values(["validation_accuracy","train_accuracy"],ascending=False).iloc[0]
    selected=models[(best.kernel,float(best.C))]
    test_pred=selected.predict(Zte)
    test_acc=accuracy_score(yte,test_pred)
    # Difficult class pairs: compute pairwise errors using one-vs-one decision scores
    # and report three pairs with largest pair-restricted error count.
    pair_rows=[]
    for a in range(10):
        for b in range(a+1,10):
            mask=np.isin(yte,[a,b]); pair_acc=accuracy_score(yte[mask],test_pred[mask])
            pair_rows.append((pair_acc,int(mask.sum()),a,b))
    hard=sorted(pair_rows)[:3]
    pair_report=[]
    for pair_rank,(pair_acc,pair_n,a,b) in enumerate(hard, start=1):
        # Binary pair model uses only the already-fixed SVM training subset.
        train_pair_mask=np.isin(ytr[chosen],[a,b])
        pair_train_X=Ztr[chosen][train_pair_mask]
        pair_train_y=ytr[chosen][train_pair_mask]
        pair_model=SVC(C=float(best.C),kernel=best.kernel,gamma="scale",
                       degree=3,cache_size=1500)
        pair_model.fit(pair_train_X,pair_train_y)
        test_pair_ids=np.where(np.isin(yte,[a,b]))[0]
        pair_scores=pair_model.decision_function(Zte[test_pair_ids])
        nearest_order=np.argsort(np.abs(pair_scores))[:20]
        near_ids=test_pair_ids[nearest_order]
        # Support-vector rows refer to positions in the fixed global SVM subset.
        pair_subset_global=chosen[train_pair_mask]
        support_global=pair_subset_global[pair_model.support_]
        np.savez(out/f"pair_{a}_{b}_boundary_analysis.npz",
                 test_near_boundary_indices=near_ids,
                 test_near_boundary_decision_scores=pair_scores[nearest_order],
                 training_support_vector_indices=support_global,
                 training_support_vectors=pair_model.support_vectors_)
        pair_report.append({"class_a":CLASS_NAMES[a],"class_b":CLASS_NAMES[b],
                            "pair_accuracy":pair_acc,"n_test":pair_n,
                            "near_boundary_test_indices":near_ids.tolist(),
                            "near_boundary_abs_decision_scores":np.abs(pair_scores[nearest_order]).tolist(),
                            "training_support_vector_count":len(support_global)})
    pd.DataFrame(pair_report).to_json(out/"difficult_pairs_boundary.json",orient="records",indent=2)
    pd.DataFrame([{"class_a":CLASS_NAMES[a],"class_b":CLASS_NAMES[b],
                   "pair_accuracy":acc,"n_test":n} for acc,n,a,b in hard]).to_csv(out/"difficult_pairs.csv",index=False)
    # Cluster: choose K by validation silhouette on a deterministic validation subset.
    # KMeans fit only on training representations; validation labels are not used.
    from sklearn.metrics import silhouette_score
    krows=[]
    val_sub=Zv[:min(3000,len(Zv))]
    for k in [10,15,20]:
        km=KMeans(n_clusters=k,random_state=SEED,n_init=10)
        km.fit(Ztr)
        val_cluster=km.predict(val_sub)
        sil=silhouette_score(val_sub,val_cluster, sample_size=min(1500,len(val_sub)),random_state=SEED)
        krows.append({"k":k,"validation_silhouette":sil})
    pd.DataFrame(krows).to_csv(out/"kmeans_validation.csv",index=False)
    best_k=max(krows,key=lambda r:r["validation_silhouette"])["k"]
    km=KMeans(n_clusters=best_k,random_state=SEED,n_init=10).fit(Ztr)
    test_clusters=km.predict(Zte)
    ari=adjusted_rand_score(yte,test_clusters); nmi=normalized_mutual_info_score(yte,test_clusters)
    contingency=confusion_matrix(yte,test_clusters,labels=list(range(best_k)))
    # Save class x cluster table (rows classes, columns cluster IDs)
    table=np.zeros((10,best_k),dtype=int)
    for y,c in zip(yte,test_clusters): table[y,c]+=1
    pd.DataFrame(table,index=CLASS_NAMES,columns=[f"cluster_{i}" for i in range(best_k)]).to_csv(out/"cluster_class_contingency.csv")
    save_json({"selected_kernel":best.kernel,"selected_C":float(best.C),
               "validation_accuracy":float(best.validation_accuracy),
               "test_accuracy":test_acc,"selected_k":best_k,"test_ARI":ari,"test_NMI":nmi,
               "svm_subset_size":len(chosen)},out/"q4_results.json")
    # Sweep plot
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for kernel in kernels:
        sub=df[df.kernel==kernel]
        axs[0].plot(sub.C,sub.validation_accuracy,marker="o",label=kernel)
        axs[1].plot(sub.C,sub.support_vector_fraction,marker="o",label=kernel)
    for ax in axs: ax.set_xscale("log"); ax.set_xlabel("C"); ax.grid(alpha=.25); ax.legend()
    axs[0].set_ylabel("Validation accuracy"); axs[1].set_ylabel("Support vector fraction")
    fig.tight_layout(); fig.savefig(out/"svm_sweep.png",dpi=180); plt.close(fig)
    print("Q4:",best.to_dict(),"test accuracy",test_acc,"ARI",ari,"NMI",nmi)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="./data"); p.add_argument("--output-dir",default="./outputs")
    p.add_argument("--batch-size",type=int,default=256); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--pca-components",type=int,default=100); p.add_argument("--subset-per-class",type=int,default=1000)
    run(p.parse_args())
