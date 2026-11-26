"""Paper text. Numbers are injected from runs/results.json via <<python-expression>> placeholders."""
from paths import ROOT, DATA  # noqa: E402
import re, json, os

R = json.load(open(ROOT + "runs/results.json"))
M = R["main"]
FIG = ROOT + "paper/figures/"
NAMES = {"pixel_lr": "Raw-pixel LR (prior report)", "forensic_lr": "Forensic features + LR", "forensic_gbdt": "Forensic features + GBDT",
         "global_cnn": "Global ResNet-50 (512\\,px)", "weak": "FF-R, image labels only", "unpaired": "FF-R, unpaired negatives",
         "rgb": "FF-R, RGB only (no SRM)", "noaug": "\\textbf{FieldForensics (ours)}", "main_s0": "FieldForensics-R (ours, degr.-aug.)",
         "fusion": "FieldForensics-R + forensic GBDT", "fusion_noaug": "FieldForensics + forensic GBDT"}
SRCN = {"cord": "CORD", "wildreceipt": "WildReceipt", "sroie": "SROIE", "xfund": "XFUND"}


def has(*ks):
    d = R
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return False
        d = d[k]
    return True


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def pc(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def pv(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def X(s):
    """Evaluate <<expr>> placeholders."""
    return re.sub(r"<<(.+?)>>", lambda m: str(eval(m.group(1), globals())), s, flags=re.S).replace("=<0.001", "<0.001")


PK = R.get("primary", "main_s0")          # primary model (clean-trained FieldForensics)
m0 = M.get(PK, {})
AUC = m0.get("auc", float("nan"))
mR = M.get("main_s0", {})                 # degradation-augmented FieldForensics-R
AUCR = mR.get("auc", float("nan"))
best_base = max((M[k]["auc"], k) for k in M if k in ("pixel_lr", "forensic_lr", "forensic_gbdt", "global_cnn")) if any(k in M for k in ("pixel_lr", "forensic_lr", "forensic_gbdt", "global_cnn")) else (float("nan"), "")


def build():
    meta = {
        "title": "Localize to Detect: Full-Resolution Deep Forensics and Compression Cues for GPT-Image-2 Field Forgeries in Receipts and Forms",
        "authors": [f"{n}<br/><i>B.Tech CSE (AIML)</i><br/><i>D. Y. Patil International University</i><br/>Akurdi, Pune, India<br/>{e}" for n, e in [
            ("Omkar Suryavanshi", "20240802073@dypiu.ac.in")]],
        "authors_tex": [r"\IEEEauthorblockN{%s}\IEEEauthorblockA{\textit{B.Tech CSE (AIML)} \\ \textit{D. Y. Patil International University}\\ Akurdi, Pune, India \\ %s}" % (n, e) for n, e in [
            ("Omkar Suryavanshi", "20240802073@dypiu.ac.in")]],
        "pdf_author": "Omkar Suryavanshi",
        "keywords": "Document forensics, AI-generated image detection, generative inpainting, tampered text localization, GPT-Image-2, AIForge-Doc, noise residual, robustness",
    }
    meta["abstract"] = X(ABSTRACT)
    blocks = []
    for b in BLOCKS():
        if b[0] in ("p",):
            blocks.append((b[0], X(b[1])))
        elif b[0] in ("list", "enum"):
            blocks.append((b[0], [X(i) for i in b[1]]))
        elif b[0] == "fig":
            if os.path.exists(b[1]):
                blocks.append((b[0], b[1], X(b[2])) + tuple(b[3:]))
        elif b[0] == "tab":
            blocks.append((b[0], X(b[1])) + tuple(b[2:]))
        else:
            blocks.append(b)
    return meta, blocks


from abstract import ABSTRACT  # noqa: E402


def BLOCKS():
    B = []
    A = B.append
    # ------------------------------------------------------------------ I. Introduction
    A(("sec", "Introduction", "sec:intro"))
    A(("p", r"""Receipts, invoices, bank statements and filled-in forms are routinely accepted as evidence in expense reimbursement, insurance claims, loan applications and know-your-customer checks. Traditional document fraud---copy--move of digits, splicing of text from another document, or font substitution in an editor---leaves traces that forensic detectors have learned to exploit, such as double-JPEG compression, resampling and inconsistent noise \cite{findit,doctamper,catnet}. Instruction-following image generators have changed this picture. A forger can now mark a field and ask a model to ``replace 1,346,000 with 2,103,717 in the same font'', and the generator synthesises the new glyphs with matching ink texture, paper colour, blur and perspective. Fig.~\ref{fig:teaser} shows such a forgery and how our detector exposes it."""))
    A(("fig", FIG + "teaser.png", r"""FieldForensics at a glance. A GPT-Image-2 forgery changes a single subtotal (1,346,000 $\rightarrow$ 2,103,717) on a CORD receipt, invisible at page scale. Our network analyses the page at native resolution and outputs a tampering map that lights up exactly on the regenerated digits (dashed: ground-truth field). The page score is the peak of the map: 1.00 for the forgery and 0.16 for the untouched original.""", "fig:teaser", True, 7.0))
    A(("p", r"""AIForge-Doc \cite{aiforge1} and its second version \cite{aiforge2} quantify how serious this is. In v2, 3,066 numeric or key fields from four public receipt and form corpora \cite{cord,wildreceipt,sroie,xfund} were rewritten by OpenAI's GPT-Image-2 and pasted back into the original page. The benchmark authors report that human inspectors in a two-alternative forced-choice study achieve an AUC of 0.501, the state-of-the-art general forensic detector TruFor \cite{trufor} 0.599, the document-specific detector DocTamper \cite{doctamper} 0.585, and GPT-Image-2 judging its own forgeries 0.532 \cite{aiforge2,forgerjudge}. Both detectors exceed 0.85 AUC on traditional tampering of the same source documents, which isolates an \emph{AI-inpainting-specific} detection gap."""))
    A(("p", r"""The present work began as a course project whose first report trained a logistic regression on down-sampled document images and compared centre, random and text-density crops. In hindsight that design could not succeed: the forged field occupies a median of 0.76\% of the page (5th--95th percentile: 0.03--2.8\%), so resizing a 2480$\times$3508 form to a thumbnail erases the evidence, and fixed crops usually miss the field altogether. This paper replaces that study with a rigorous re-design built on one principle: \emph{to decide whether a document has been forged, first find where it could have been forged, at the resolution at which the forgery was made}."""))
    A(("p", r"""Our contributions are as follows:"""))
    A(("list", [
        r"\textbf{A leak-free, paired evaluation protocol for AIForge-Doc v2.} We verify that every forged image is pixel-identical to its authentic counterpart outside the mask, remove 31 duplicated source documents (6 of which straddle the official train/test split), and score both versions of each of the 606 remaining test documents, so that classifiers cannot exploit document identity, layout or source.",
        r"\textbf{FieldForensics, a localize-to-detect architecture.} A U-Net \cite{unet} with a ResNet-34 \cite{resnet} encoder fuses RGB with a fixed SRM noise residual \cite{srm,rgbn}, keeps a full-resolution skip path for stroke-level detail, and is trained jointly for pixel localisation and image-level detection using field-centred crops with paired hard negatives.",
        r"\textbf{A broad empirical study} comparing classical and deep baselines, ablating the main design choices with paired bootstrap significance tests, and measuring per-source behaviour, sensitivity to forged-region size<<', robustness to realistic degradations and generalisation to unseen document sources' if R.get('loso') else ' and robustness to realistic degradations'>>.",
        r"\textbf{An analysis of the cues.} Block-wise compression statistics are a surprisingly strong baseline that exploits compression-history inconsistencies. A network trained on clean data learns these cues by itself, whereas degradation augmentation suppresses them. The result is a clear trade-off between clean accuracy and robustness to re-compression, which we quantify, together with a document-type effect that masquerades as a forged-area effect."]))

    # ------------------------------------------------------------------ II. Related work
    A(("sec", "Related Work", "sec:rel"))
    A(("subsec", "Document tampering detection"))
    A(("p", r"""Early document-forensics work relied on hand-crafted cues such as character alignment, font consistency and printer signatures; the ICPR 2018 \emph{Find it!} contest \cite{findit} established receipts as a benchmark domain. Deep approaches followed general image-manipulation detection: RGB-N \cite{rgbn} paired an RGB stream with an SRM noise stream \cite{srm}, ManTra-Net \cite{mantra} learned anomalous local features, MVSS-Net \cite{mvss} combined edge and noise supervision, CAT-Net \cite{catnet} exploited JPEG DCT statistics, and TruFor \cite{trufor} learned a camera-noise fingerprint (Noiseprint++) with a reliability map. DocTamper \cite{doctamper} adapted these ideas to text images using frequency-domain features and multi-JPEG training. These detectors were designed for splicing, copy--move and classical inpainting, and AIForge-Doc shows that they transfer poorly to generative edits \cite{aiforge1,aiforge2}."""))
    A(("subsec", "AI-generated image detection"))
    A(("p", r"""Fully synthetic images can often be detected from generator fingerprints in the frequency spectrum \cite{wang2020,corvi} or from the feature space of large pretrained encoders \cite{ojha}, although generalisation to unseen generators and post-processing remains open \cite{li2025}. Deep inpainting of natural images has been localised with high-pass fully convolutional networks \cite{hpfcn}. Fu \emph{et al.} \cite{fu2026} showed that detectors trained on natural images do not carry over to documents, whose statistics are dominated by text, flat backgrounds and scanning artefacts. Field-level forgeries are the hardest case: the edited region is tiny and the remaining $>$99\% of pixels are genuine, so image-level fingerprints are diluted by authentic content."""))
    A(("subsec", "Benchmarks for AI-forged documents"))
    A(("p", r"""AIForge-Doc v1 \cite{aiforge1} introduced 4,062 forgery specifications executed with Gemini 2.5 Flash Image and Ideogram v2; v2 \cite{aiforge2} re-executes the same specifications with GPT-Image-2 and, crucially, releases the authentic source of every forgery, pixel-aligned, together with the tampering mask. Its authors evaluate detectors zero-shot; to the best of our knowledge we report the first supervised detectors and localisers trained and evaluated on the v2 split under a leak-free protocol."""))

    # ------------------------------------------------------------------ III. Data & protocol
    A(("sec", "Dataset and Evaluation Protocol", "sec:data"))
    A(("subsec", "AIForge-Doc v2"))
    A(("p", r"""Each forgery is produced in four steps \cite{aiforge2}: a numeric or key field (price, total, date, telephone number, document ID or form answer) is selected from the source annotation and a plausible replacement string is generated; the field box is enlarged by 50\% to give the generator context; GPT-Image-2 inpaints the box under a prompt that fixes the target string and asks for typography matching; and only the exact field box is pasted back into the full-resolution page, which is saved as a metadata-free PNG. Fig.~\ref{fig:examples} shows one example per source. Table~\ref{tab:data} summarises the composition after our de-duplication."""))
    A(("fig", FIG + "examples.png", r"""One forged document per source. Left: full page with the forged field outlined. Middle: authentic and GPT-Image-2 versions of the field at native resolution. Right: absolute difference, which is non-zero only inside the field. The edits are visually seamless, and the forged area is a tiny fraction of the page.""", "fig:examples", True, 7.0))
    A(("tab", r"Composition of the de-duplicated AIForge-Doc v2 splits (documents; each contributes one authentic and one forged image)", "tab:data",
       ["Source", "Type / orig. format", "Train", "Val", "Test", "Median size (px)"],
       [["CORD v2 \\cite{cord}", "receipt / PNG", "707", "78", "196", "864$\\times$1296"],
        ["WildReceipt \\cite{wildreceipt}", "receipt / JPEG", "962", "107", "267", "348$\\times$450"],
        ["SROIE \\cite{sroie}", "receipt / JPEG", "216", "24", "60", "623$\\times$1471"],
        ["XFUND \\cite{xfund}", "form / PNG", "301", "34", "83", "2480$\\times$3508"],
        ["Total", "", "2,186", "243", "606", ""]], False, [0.27, 0.2, 0.1, 0.09, 0.1, 0.24]))
    A(("p", r"""Fig.~\ref{fig:stats} visualises three properties that drive our design. WildReceipt and CORD dominate the data (a). The forged region is tiny: its area spans three orders of magnitude, with XFUND fields covering only about 0.01--0.1\% of a page (b). Prices, totals and phone numbers, i.e. the fields a fraudster would actually target, make up most edits (c)."""))
    A(("fig", FIG + "dataset_stats.png", r"""AIForge-Doc v2 after de-duplication. (a) Documents per source and split. (b) Distribution of the forged area as a percentage of the page (log scale), by source; the dashed line marks the median of 0.76\%. (c) Types of forged fields.""", "fig:stats", True, 7.0))
    A(("subsec", "Integrity audit and de-duplication"))
    A(("p", r"""Before any modelling we verified three properties programmatically. (i) For all 3,066 pairs the forged and authentic images are bit-identical outside the tampering mask, and each mask is exactly the axis-aligned field rectangle. (ii) The authentic SHA-256 hashes reveal 25 groups of byte-identical source images, i.e. the same receipt occurring under two different source identifiers. Six of these groups cross the official train/test boundary, which would let a model memorise a test document during training. We therefore drop the test copy of every cross-split duplicate and keep only one copy of within-split duplicates, removing 31 documents in total. (iii) From the remaining training documents we hold out a source-stratified 10\% validation set. The final split has 2,186 training, 243 validation and 606 test documents (Table~\ref{tab:data})."""))
    A(("subsec", "Paired protocol and metrics"))
    A(("p", r"""Every test document contributes both its authentic and its forged image (1,212 test images, perfectly balanced). Because the two images of a pair share source, size, layout, language and content except for one field, any score difference must come from the forged region itself. We report the area under the ROC curve (AUC) with 95\% confidence intervals from 2,000 bootstrap resamples of \emph{documents} (pairs are resampled together) \cite{bootstrap}; accuracy, precision, recall and F1 at the fixed decision threshold 0.5; the true-positive rate at 5\% false-positive rate (TPR@5\%FPR), an operating point relevant to fraud screening; and the \emph{paired accuracy}, the fraction of documents whose forged version receives a higher score than its authentic version. Paired accuracy is the two-alternative forced-choice (2AFC) metric used for the human study in \cite{aiforge2}. For localisation we report pixel F1 and IoU of the thresholded map on forged images, and the \emph{peak-hit rate}: how often the most suspicious location in the page lies inside the forged field. Differences between systems are tested with a paired bootstrap over documents."""))

    # ------------------------------------------------------------------ IV. Method
    A(("sec", "Method: FieldForensics", "sec:method"))
    A(("fig", FIG + "method.png", r"""Overview of FieldForensics. The document is processed at native resolution. RGB and a fixed SRM noise residual are fused at the stem of a ResNet-34 U-Net with an additional full-resolution skip path. The network predicts a tampering-probability map, and the document score is the maximum of the smoothed map. Training uses field-centred crops with paired authentic negatives and a joint pixel/image loss.""", "fig:method", True, 6.6))
    A(("subsec", "Problem formulation"))
    A(("p", r"""Given a document image $I \in \{0,\ldots,255\}^{H\times W\times 3}$, the goal is to output an image-level score $s(I)\in[0,1]$ that is high if any region was regenerated, together with a map $P\in[0,1]^{H\times W}$ that localises the regenerated pixels. Training provides, for each document, the authentic image $I^{a}$, the forged image $I^{f}$ and the field mask $M$. Since the forgery is local, we cast detection as localisation followed by pooling, in the spirit of multiple-instance learning \cite{mil}: a document is forged if and only if at least one location is forged."""))
    A(("subsec", "Architecture"))
    A(("p", r"""\emph{Input streams.} Generative inpainting reproduces the look of the glyphs well but not the sensor, print and compression noise of the surrounding scan \cite{hpfcn}. Besides the ImageNet-normalised RGB channels we therefore compute three fixed SRM high-pass residuals \cite{srm,rgbn} of the luminance $Y$: a second-order 3$\times$3 kernel $K_1$, the 5$\times$5 ``KV'' kernel $K_2$ and a horizontal second-order kernel $K_3$, truncated and normalised as"""))
    A(("eq", r"N_i = \frac{1}{3}\,\mathrm{clip}\left(K_i * Y,\,-3,\,3\right),\qquad i=1,2,3.", "eq:srm"))
    A(("p", r"""RGB and noise are concatenated into a 6-channel input. The first convolution of an ImageNet-pretrained \cite{imagenet} ResNet-34 \cite{resnet} is extended to six channels: the RGB filters are copied and the noise filters are initialised with half the channel-mean of the RGB filters."""))
    A(("p", r"""\emph{Decoder.} A U-Net decoder \cite{unet} upsamples the encoder features (strides 32 down to 2) through four skip connections. Because typical forged glyph strokes are only 2--4 pixels wide, we add a fifth, \emph{full-resolution} skip branch (two 3$\times$3 convolutions with 32 channels on the raw 6-channel input), so that the final prediction can use stroke-level evidence instead of interpolating from stride 2. A 1$\times$1 convolution produces the logit map $Z$, and $P=\sigma(Z)$. The network has 24.5\,M parameters and is fully convolutional, so it runs on pages of any size without resizing."""))
    A(("subsec", "Field-centred sampling with paired hard negatives"))
    A(("p", r"""Training uses $384\times384$ crops at native resolution, sampled online. For each crop a document and a class (authentic or forged, $p=0.5$) are drawn. With probability 0.7 the crop window is placed so that it contains the field box at a random offset, otherwise it is uniform over the page. The key detail is that \emph{authentic crops are drawn from the same field-centred distribution as forged crops}: the negative example is the original field at exactly the same position, with the same digits' neighbours, font and background. The only difference between a positive and its paired negative is therefore who rendered the glyphs. This rules out shortcuts such as ``numeric fields look suspicious'' \cite{geirhos} and forces the network to learn generation traces. The \emph{unpaired} ablation instead samples authentic crops uniformly over the page."""))
    A(("subsec", "Training objective"))
    A(("p", r"""With valid-pixel mask $V$ (crops of small pages are padded) and crop label $y=\max M$, the loss combines a pixel term, a Dice term \cite{vnet} computed on forged crops, and an image term on a top-$k$ pooled logit:"""))
    A(("eq", r"\mathcal{L} = \mathrm{BCE}_{w}(Z, M; V) + \mathrm{Dice}(P, M) + \frac{1}{2}\,\mathrm{BCE}\big(\mathrm{top}k(Z),\, y\big)", "eq:loss"))
    A(("p", r"""Here $\mathrm{BCE}_w$ is the binary cross-entropy averaged over valid pixels with a positive weight of 5, and $\mathrm{top}k$ averages the $k=256$ largest logits. The weakly supervised ablation keeps only the image term, i.e. it is trained from image labels without masks."""))
    A(("subsec", "Inference and image score"))
    A(("p", r"""At test time the full page is processed in $1024\times1024$ tiles with 128-pixel overlap, without any resizing. The predicted map is averaged over overlaps, smoothed with a $9\times9$ box filter $\mathcal{B}$ to suppress isolated pixels, and its maximum gives the document score:"""))
    A(("eq", r"s(I) = \max_{(u,v)}\; (\mathcal{B} * P)(u,v).", "eq:score"))
    A(("p", r"""The location of this maximum is the system's answer to ``where was this document edited''; Section~\ref{sec:results} compares it with alternative pooling rules."""))
    A(("subsec", "Degradation augmentation (FieldForensics-R)"))
    A(("p", r"""Real documents are re-compressed, rescaled and blurred when they are photographed, shared or uploaded. We therefore also train a robustness-oriented configuration, \emph{FieldForensics-R}, that differs from FieldForensics only in its training augmentation. With probability 0.5 a training crop is JPEG-compressed at a quality drawn from $[50,95]$. To mimic re-compression of a whole page rather than of an arbitrary crop, the crop is first extended to the image's own $8\times8$ block grid, compressed, and then cut, so that the block boundaries stay aligned with the page. In addition, crops are rescaled by a factor in $[0.5,1]$ ($p=0.2$), blurred ($\sigma\in[0.3,1]$, $p=0.1$) or corrupted with Gaussian noise ($\sigma\in[1,5]$, $p=0.1$)."""))

    A(("subsec", "Late fusion with compression statistics"))
    A(("p", r"""Besides generator traces, a pasted patch also leaves a \emph{compression-history} trace when the page was JPEG-compressed before the edit and the patch was not. To test whether the network captures this cue, we model it explicitly with 20 block-wise statistics (Section~\ref{sec:setup}): maxima, percentiles and max/median ratios over 32$\times$32 blocks of error-level analysis \cite{ela}, a high-pass residual and an 8$\times$8 JPEG blockiness map, classified with gradient-boosted trees. A late fusion averages network and GBDT probabilities with fixed, equal weights, $s_{\mathrm{fuse}} = \frac{1}{2}(s_{\mathrm{net}} + s_{\mathrm{GBDT}})$. The weights are not tuned, so the fusion uses no test information. If the fusion improves on the network, the network has missed the compression cue."""))

    # ------------------------------------------------------------------ V. Setup
    A(("sec", "Experimental Setup", "sec:setup"))
    A(("subsec", "Baselines"))
    A(("p", r"""\emph{Raw-pixel LR} reproduces the baseline of our earlier report: the grey-scale page is resized to $64\times64$ and classified with an $\ell_2$-regularised logistic regression. \emph{Forensic features} are 20 hand-crafted statistics computed at native resolution: block-wise (32$\times$32) mean, spread, maximum, 99th percentile and max/median ratios of error-level analysis (ELA, quality 90) \cite{ela}, of a high-pass noise residual, and of a JPEG 8$\times$8 blockiness map. They are classified with logistic regression or gradient-boosted trees (GBDT) \cite{sklearn}. The \emph{Global ResNet-50} fine-tunes an ImageNet ResNet-50 \cite{resnet} on whole pages resized to $512\times512$, with the same JPEG augmentation. These baselines represent the ``classify the whole page'' paradigm. Published zero-shot results of TruFor \cite{trufor}, DocTamper \cite{doctamper}, GPT-Image-2 self-judgement and humans \cite{aiforge2} are listed for reference. They were obtained on all 6,132 images rather than our test split, so they are indicative rather than directly comparable."""))
    A(("subsec", "Implementation details"))
    A(("p", r"""All models are implemented in PyTorch \cite{pytorch} and trained on a single laptop GPU (NVIDIA RTX 3060, 6\,GB) with mixed precision. FieldForensics, FieldForensics-R and every ablation are trained for the same budget of 1,500 iterations with batch size 8 (12,000 crops), using AdamW \cite{adamw} (learning rate $3\times10^{-4}$, weight decay $10^{-4}$, 300-iteration warm-up followed by cosine decay) and gradient clipping at 5. The final checkpoint is evaluated, and no test-set information is used for model selection or thresholds. The global CNN is trained for 4 epochs at batch size 16 and selected on validation AUC. All splits, seeds and per-image predictions are saved with the code."""))

    # ------------------------------------------------------------------ VI. Results
    A(("sec", "Results", "sec:results"))
    B.extend(RESULTS())
    A(("sec", "Discussion", "sec:disc"))
    B.extend(DISCUSSION())
    A(("sec", "Limitations and Future Work", "sec:lim"))
    B.extend(LIMITS())
    A(("sec", "Conclusion", "sec:concl"))
    B.extend(CONCL())
    A(("sec", "*Acknowledgment"))
    A(("p", r"""The author thanks the creators of AIForge-Doc and of the CORD, WildReceipt, SROIE and XFUND corpora for releasing their data for research use."""))
    return B


from paper_results import RESULTS, DISCUSSION, LIMITS, CONCL  # noqa: E402
