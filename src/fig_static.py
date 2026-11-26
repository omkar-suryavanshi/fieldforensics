from paths import ROOT, DATA  # noqa: E402
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from common import load_splits, load_image
FIG=ROOT + "paper/figures/"
plt.rcParams.update({"font.family":"serif","font.size":7,"figure.dpi":220,"savefig.bbox":"tight"})
recs=load_splits(); te=[r for r in recs if r["split"]=="testing"]
pick=[[r for r in te if r["source"]==s][1] for s in ["cord","wildreceipt","sroie","xfund"]]
fig,axes=plt.subplots(4,4,figsize=(7.0,4.6))
for i,r in enumerate(pick):
    a,_=load_image(r,0); f,m=load_image(r,1)
    x1,y1,x2,y2=r["mbbox"]; pad=max(60,(x2-x1)//2)
    X1,Y1,X2,Y2=max(0,x1-pad),max(0,y1-pad//2),min(a.shape[1],x2+pad),min(a.shape[0],y2+pad//2)
    axes[i,0].imshow(f); axes[i,0].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec="r",lw=1))
    axes[i,1].imshow(a[Y1:Y2,X1:X2]); axes[i,2].imshow(f[Y1:Y2,X1:X2])
    d=np.abs(f.astype(int)-a.astype(int)).sum(2)[Y1:Y2,X1:X2]; axes[i,3].imshow(d,cmap="magma")
    axes[i,0].set_ylabel(f"{r['source'].upper()}\n{a.shape[1]}×{a.shape[0]}",fontsize=7)
    axes[i,1].set_title(f"authentic: field '{r['field'][:22]}'" if i==0 else "",fontsize=7)
for j,t in enumerate(["Forged document (field in red)","Authentic field (zoom)","GPT-Image-2 forged field (zoom)","|forged − authentic|"]):
    axes[0,j].set_title(t,fontsize=7)
for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(pad=0.3); fig.savefig(FIG+"examples.png"); plt.close(fig)

# method diagram
fig,ax=plt.subplots(figsize=(7.0,2.1)); ax.set_xlim(-2,104); ax.set_ylim(-1,29); ax.axis("off")
def box(x,y,w,h,t,c,fs=6.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5",fc=c,ec="k",lw=0.6)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(x1,y1,x2,y2): ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="->",lw=0.7))
N=chr(10)
box(0,16,15,10,N.join(["Document at","native resolution","(no resizing)"]),"#e3f2fd")
box(0,1,15,10,N.join(["Train: 384px crops,","field-centred,","paired auth./forged"]),"#fff3e0",5.8)
box(20,20,14,6,N.join(["RGB","(ImageNet norm.)"]),"#e8f5e9"); box(20,10,14,6,N.join(["SRM noise","residual (3 ch)"]),"#e8f5e9")
box(39,10,16,16,N.join(["ResNet-34","encoder","(6-ch stem)"]),"#ede7f6"); box(60,10,16,16,N.join(["U-Net decoder","+ full-res skip","(stroke detail)"]),"#ede7f6")
box(81,19,21,7,N.join(["Tamper-probability","map (localisation)"]),"#ffebee"); box(81,7,21,7,N.join(["Image score: max of","9x9-smoothed map"]),"#ffebee")
box(25,0,51,5,"Loss = BCE + Dice (pixels) + 0.5 BCE (top-k pooled logit, image)","#fffde7",5.8)
arr(15,21,20,23); arr(15,21,20,13); arr(34,23,39,20); arr(34,13,39,15); arr(55,18,60,18); arr(76,20,81,22); arr(91.5,19,91.5,14); arr(7.5,11,7.5,16)
fig.savefig(FIG+"method.png"); plt.close(fig)
print("ok")
