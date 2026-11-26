from paths import ROOT, DATA  # noqa: E402
import json, csv, collections, numpy as np
from PIL import Image
R=DATA
meta=[json.loads(l) for l in open(R+"metadata.jsonl",encoding="utf8")]
sm=list(csv.DictReader(open(R+"source_mapping.csv",encoding="utf8")))
print(len(meta),len(sm))
print(collections.Counter((m['split'],m['source_dataset']) for m in meta))
# duplicates of authentic hashes
h=collections.defaultdict(list)
for r in sm: h[r['authentic_sha256']].append((r['split'],r['new_id'],r['image_id']))
dups=[v for v in h.values() if len(v)>1]
print("dup hash groups",len(dups),"cross-split",sum(1 for v in dups if len({x[0] for x in v})>1))
print(dups[:5])
img=collections.Counter(m['image_id'] for m in meta); print("dup image_id",sum(1 for v in img.values() if v>1))
W=np.array([int(r['width']) for r in sm]);H=np.array([int(r['height']) for r in sm])
print("W",np.percentile(W,[0,5,50,95,100]),"H",np.percentile(H,[0,5,50,95,100]))
for src in ['cord','wildreceipt','sroie','xfund']:
    rs=[r for r in sm if r['source_dataset']==src]; print(src,collections.Counter(r['orig_format'] for r in rs), np.median([int(r['width']) for r in rs]),np.median([int(r['height']) for r in rs]))
area=np.array([(m['bbox_xyxy'][2]-m['bbox_xyxy'][0]+1)*(m['bbox_xyxy'][3]-m['bbox_xyxy'][1]+1) for m in meta])
frac=area/(W*H)
print("bbox area px",np.percentile(area,[5,50,95]),"frac %",np.percentile(frac*100,[5,50,95]))
print(collections.Counter(m['field_name'].split('.')[0] for m in meta).most_common(12))
