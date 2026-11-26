from paths import ROOT, DATA  # noqa: E402
import json,numpy as np, random
from PIL import Image
R=DATA
meta=[json.loads(l) for l in open(R+"metadata.jsonl",encoding="utf8")]
random.seed(0)
for m in random.sample(meta,12):
    d="TrainingSet" if m['split']=='training' else "TestingSet"
    f=np.array(Image.open(R+m['image']).convert('RGB')).astype(int); a=np.array(Image.open(R+f"{d}/authentic/{m['new_id']}.png").convert('RGB')).astype(int)
    k=np.array(Image.open(R+m['mask']).convert('L'))
    ys,xs=np.where(k>127); rect=(k>127).sum()==(np.ptp(ys)+1)*(np.ptp(xs)+1)
    diff=np.abs(f-a).sum(2)
    print(m['source_dataset'],f.shape,"mask bbox",xs.min(),ys.min(),xs.max(),ys.max(),"meta",m['bbox_xyxy'],"rect",rect,"diff outside",diff[k<=127].max(),"inside mean",diff[k>127].mean().round(1), "frac changed in mask",(diff[k>127]>0).mean().round(3), np.unique(k))
