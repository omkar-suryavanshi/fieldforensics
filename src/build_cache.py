"""Build deduplicated splits + fast npy cache. Forged image = authentic with bbox patch replaced (verified)."""
from paths import ROOT, DATA  # noqa: E402
import json, csv, os, random, collections, numpy as np
from PIL import Image
from multiprocessing import Pool
R=DATA; C=ROOT + "cache/"
os.makedirs(C+"img",exist_ok=True)
def work(r):
    out=C+"img/"+r['uid']
    d=r['dir']
    a=np.array(Image.open(R+f"{d}/authentic/{r['new_id']}.png").convert('RGB'))
    f=np.array(Image.open(R+f"{d}/images/{r['new_id']}.png").convert('RGB'))
    k=np.array(Image.open(R+f"{d}/masks/{r['new_id']}.png").convert('L'))>127
    ys,xs=np.where(k); x1,y1,x2,y2=xs.min(),ys.min(),xs.max(),ys.max()
    g=f.copy(); g[y1:y2+1,x1:x2+1]=a[y1:y2+1,x1:x2+1]
    ok=(g==a).all() and f.shape==a.shape
    np.save(out+"_auth.npy",a)
    if ok: np.save(out+"_patch.npy",f[y1:y2+1,x1:x2+1])
    else: np.save(out+"_patch.npy",f); np.save(out+"_mask.npy",k)
    return r['uid'],[int(x1),int(y1),int(x2),int(y2)],bool(ok),float(k.sum()/k.size)
if __name__=="__main__":
    meta=[json.loads(l) for l in open(R+"metadata.jsonl",encoding="utf8")]
    sm={r['spec_id']:r for r in csv.DictReader(open(R+"source_mapping.csv",encoding="utf8"))}
    recs=[]
    for m in meta:
        s=sm[m['spec_id']]
        recs.append(dict(uid=m['spec_id'],dir='TrainingSet' if m['split']=='training' else 'TestingSet',split=m['split'],new_id=m['new_id'],source=m['source_dataset'],doc_type=m['doc_type'],
            lang=m['language'],field=m['field_name'],bbox=m['bbox_xyxy'],w=int(s['width']),h=int(s['height']),fmt=s['orig_format'],sha=s['authentic_sha256']))
    groups=collections.defaultdict(list)
    for r in recs: groups[r['sha']].append(r)
    drop=set()
    for g in groups.values():
        if len(g)>1:
            tr=[r for r in g if r['split']=='training']; te=[r for r in g if r['split']=='testing']
            if tr: drop|={r['uid'] for r in te}; drop|={r['uid'] for r in tr[1:]}
            else: drop|={r['uid'] for r in te[1:]}
    keep=[r for r in recs if r['uid'] not in drop]
    random.seed(0)
    train=[r for r in keep if r['split']=='training']; test=[r for r in keep if r['split']=='testing']
    for src in ['cord','wildreceipt','sroie','xfund']:
        s=[r for r in train if r['source']==src]; random.shuffle(s)
        for r in s[:round(0.1*len(s))]: r['split']='val'
    for r in keep:
        if r['split']=='training': r['split']='train'
    json.dump(keep,open(C+"splits.json","w"),indent=0)
    print("dropped",len(drop),collections.Counter((r['split'],r['source']) for r in keep), collections.Counter(r['split'] for r in keep))
    with Pool(12) as p: res=p.map(work,keep,chunksize=8)
    info={u:(b,ok,fr) for u,b,ok,fr in res}
    for r in keep: r['mbbox'],r['aligned'],r['mask_frac']=info[r['uid']]
    json.dump(keep,open(C+"splits.json","w"),indent=0)
    print("not aligned:",[(r['uid'],r['source']) for r in keep if not r['aligned']])
    print("done")
