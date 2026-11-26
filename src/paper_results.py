import re
from paper_content import R, M, FIG, NAMES, SRCN, has, f3, pc, ci, pv, m0, AUC, PK, mR, AUCR


def X(s):
    import sys
    loc = sys._getframe(1).f_locals
    return re.sub(r"<<(.+?)>>", lambda m: str(eval(m.group(1), globals(), loc)), s, flags=re.S).replace("=<0.001", "<0.001")


SRC = ["cord", "wildreceipt", "sroie", "xfund"]
COND_NAMES = {"jpeg90": "JPEG quality 90", "jpeg75": "JPEG quality 75", "jpeg50": "JPEG quality 50", "scale0.75": "down-scaling to 75\\%",
              "scale0.5": "down-scaling to 50\\%", "blur1.0": "Gaussian blur", "noise5": "additive noise",
              "social": "the messaging-app re-share (75\\% scale + JPEG 80)"}


def andjoin(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def sig(n, ref="signif"):
    s = R.get(ref, {}).get(n)
    if not s:
        return ""
    p = s["p"]
    if p < 0.05:
        return "(paired bootstrap $p" + ("<0.001" if p < 0.001 else "=" + pv(p)) + "$)"
    return f"($p={pv(p)}$, not significant)"


def rob(n, c):
    return R.get("robust", {}).get(n, {}).get(c, {}).get("auc")


def RESULTS():
    B = []
    A = B.append
    # ------------------------------------------------------------ main table
    order = [k for k in ["pixel_lr", "global_cnn", "forensic_lr", "forensic_gbdt", "main_s0", "fusion", "fusion_noaug", PK] if k in M]
    order = list(dict.fromkeys(order))
    rows = [["Human 2AFC \\cite{aiforge2}$^\\dagger$", "0.501", "--", "--", "--", "--", "--"],
            ["GPT-Image-2 self-judge \\cite{forgerjudge}$^\\dagger$", "0.532", "--", "--", "--", "--", "--"],
            ["DocTamper \\cite{doctamper}$^\\dagger$", "0.585", "--", "--", "--", "--", "--"],
            ["TruFor \\cite{trufor}$^\\dagger$", "0.599", "--", "--", "--", "--", "--"]]
    for k in order:
        m = M[k]
        rows.append([NAMES[k], f"{m['auc']:.3f} {ci(m['auc_ci'])}", pc(m["acc"]), pc(m["f1"]), pc(m["tpr_at_fpr5"]), pc(m["pair_acc"]),
                     f3(m.get("pix_f1")) if k in ("main_s0", "noaug", PK) else "--"])
    A(("p", X(r"""Table~\ref{tab:main} compares all systems on the 1,212 paired test images, and Fig.~\ref{fig:roc} shows the ROC curves. We report two configurations of our network: \emph{FieldForensics}, trained on clean crops, and \emph{FieldForensics-R}, trained with the degradation augmentation of Section~\ref{sec:method}-F. They are otherwise identical.""")))
    A(("tab", "Image-level detection and localisation on the de-duplicated paired test set (606 authentic + 606 forged images). Acc./F1 at threshold 0.5. $^\\dagger$Published zero-shot results on all 6,132 images \\cite{aiforge2}; not directly comparable.",
       "tab:main", ["Method", "AUC [95\\% CI]", "Acc. (\\%)", "F1 (\\%)", "TPR@5\\%FPR", "Paired acc. (\\%)", "Pixel F1"], rows, True,
       [0.29, 0.18, 0.09, 0.09, 0.11, 0.13, 0.11], (len(rows) - 1,)))
    t = []
    t.append(r"""The raw-pixel logistic regression of our earlier report is at chance (AUC <<f3(M['pixel_lr']['auc'])>>, paired accuracy <<pc(M['pixel_lr']['pair_acc'])>>\%): after down-sampling to $64\times64$, the two versions of a page differ by a fraction of one pixel.""")
    if "global_cnn" in M:
        t.append(r"""The ImageNet ResNet-50 that classifies whole pages at $512\times512$ is also at chance (<<f3(M['global_cnn']['auc'])>>), because resizing discards exactly the high-frequency evidence that separates generated from scanned glyphs.""")
    t.append(r"""Hand-crafted forensic statistics computed at native resolution are a surprisingly strong baseline (AUC <<f3(M['forensic_lr']['auc'])>> with LR and <<f3(M['forensic_gbdt']['auc'])>> with GBDT). Their block-wise \emph{maxima} of ELA, noise and blockiness respond to a single anomalous region, so they too follow the localize-to-detect principle, only with fixed features.""")
    if PK != "main_s0":
        t.append(r"""FieldForensics is the best system by a wide margin: AUC <<f3(AUC)>> (95\% CI <<ci(m0['auc_ci'])>>), <<'%+.3f' % R['signif_primary']['forensic_gbdt']['diff']>> over the forensic GBDT <<sig('forensic_gbdt', 'signif_primary')>>. At a 5\% false-positive rate it flags <<pc(m0['tpr_at_fpr5'])>>\% of forgeries. In the paired 2AFC setting it identifies the forged version of a document <<pc(m0['pair_acc'])>>\% of the time, where human inspectors score 50.1\% \cite{aiforge2}. It also says \emph{where} the forgery is: the peak of its map lies inside the forged field for <<pc(m0['peak_hit'])>>\% of forged images, and the thresholded map reaches a pixel F1 of <<f3(m0['pix_f1'])>> (IoU <<f3(m0['pix_iou'])>>).""")
        t.append(r"""FieldForensics-R pays for its degradation augmentation with a clean AUC of <<f3(AUCR)>> <<sig('main_s0', 'signif_primary')>> and pixel F1 <<f3(mR['pix_f1'])>>.""")
        if "fusion" in M and "fusion_noaug" in M:
            t.append(r"""Averaging network and GBDT probabilities with fixed equal weights helps FieldForensics-R substantially (AUC <<f3(M['fusion']['auc'])>>, <<'%+.3f' % R['signif_fusion']['main_s0']['diff']>>, $p<<'<0.001' if R['signif_fusion']['main_s0']['p'] < 0.001 else '=' + pv(R['signif_fusion']['main_s0']['p'])>>$) but not FieldForensics (<<f3(M['fusion_noaug']['auc'])>> vs. <<f3(AUC)>>). Section~\ref{sec:results}-A explains why: the clean-trained network already captures the compression cues that the GBDT relies on.""")
    A(("p", X(" ".join(t))))
    A(("fig", FIG + "roc.png", "ROC curves on the paired test set (AUC in parentheses). FieldForensics-R is the degradation-augmented configuration.", "fig:roc", False, 3.3))
    A(("p", X(r"""Fig.~\ref{fig:scores} shows why the AUCs differ so much. The prior pixel baseline gives every page almost the same score. The forensic GBDT and FieldForensics-R produce broad, overlapping distributions. FieldForensics is nearly binary: most authentic pages score below 0.2, and most forgeries score close to 1.""")))
    A(("fig", FIG + "score_dist.png", "Distribution of document scores for authentic (blue) and forged (red) test pages. From left to right the classes separate more cleanly, ending with FieldForensics, which assigns near-binary scores.", "fig:scores", True, 7.0))

    # ------------------------------------------------------------ per source
    if has("per_source_main"):
        ps, pso = R["per_source_main"], R["per_source"]
        A(("subsec", "Per-source analysis and complementary cues"))
        rows = [[SRCN[s] + (" (JPEG)" if s in ("wildreceipt", "sroie") else " (PNG)"), f"{ps[s]['auc']:.3f} {ci(ps[s]['auc_ci'])}", f3(pso.get("main_s0", {}).get(s)),
                 f3(pso.get("forensic_gbdt", {}).get(s)), f3(pso.get("fusion", {}).get(s)), pc(ps[s]["pair_acc"]), f3(ps[s]["pix_f1"]), pc(ps[s]["peak_hit"])] for s in SRC]
        A(("tab", "Per-source results (test split; storage format of the source release in parentheses). AUC of FieldForensics (FF, with 95\\% CI), FieldForensics-R (FF-R), the forensic-statistics GBDT and the FF-R + GBDT fusion; paired accuracy, pixel F1 and peak-hit rate of FF", "tab:source",
           ["Source", "FF AUC [95\\% CI]", "FF-R", "GBDT", "FF-R + GBDT", "Paired (\\%)", "Pix. F1", "Peak hit (\\%)"], rows, True,
           [0.17, 0.2, 0.08, 0.08, 0.11, 0.12, 0.1, 0.14]))
        A(("p", X(SOURCE_NOTE())))
        A(("fig", FIG + "per_source.png", "Per-source AUC of the four strongest systems (values above bars). FieldForensics leads on every source; the hand-crafted compression statistics (GBDT) are competitive only on the JPEG-distributed receipts.", "fig:persrc", False, 3.3))
        if has("size"):
            A(("p", X(r"""Fig.~\ref{fig:size} groups the test documents into quartiles of forged area relative to the page (the smallest quartile covers <<'%.2f' % R['size']['0']['hi']>>\% of the page or less). """ + SIZE_NOTE())))
            A(("fig", FIG + "size.png", "Detection AUC and pixel F1 of FieldForensics against the forged area (quartiles of the percentage of page pixels that were regenerated). Labels give the number of XFUND documents per quartile.", "fig:size", False, 3.3))

    # ------------------------------------------------------------ ablations (on the augmented configuration)
    abl = [k for k in ["rgb", "unpaired", "weak"] if k in M]
    if abl:
        A(("subsec", "Ablation study"))
        lbl = {"rgb": "-- SRM noise stream (RGB only)", "unpaired": "-- paired hard negatives (uniform auth. crops)", "weak": "-- pixel supervision (image labels only)",
               "noaug": "-- degradation augmentation (= FieldForensics)"}
        rows = [["FieldForensics-R (full)", f3(AUCR), "--", "--", pc(mR["pair_acc"]), f3(mR.get("pix_f1")), pc(mR["tpr_at_fpr5"])]]
        for k in abl + (["noaug"] if "noaug" in M else []):
            s = R["signif"][k]
            rows.append([lbl[k], f3(M[k]["auc"]), f"{-s['diff']:+.3f}", pv(s["p"]), pc(M[k]["pair_acc"]), f3(M[k].get("pix_f1")), pc(M[k]["tpr_at_fpr5"])])
        A(("tab", "Ablations of FieldForensics-R (same training budget; ΔAUC relative to the full augmented model; $p$ from a paired bootstrap over test documents)", "tab:abl",
           ["Variant", "AUC", "ΔAUC", "$p$", "Paired (\\%)", "Pix. F1", "TPR@5\\%"], rows, False, [0.36, 0.1, 0.1, 0.1, 0.12, 0.1, 0.12]))
        A(("p", X(r"""The ablations were run on the augmented configuration, which was our initial default (Table~\ref{tab:abl}). """ + ABL_NOTE(abl))))
    if has("aggregation"):
        A(("p", X(r"""\emph{Pooling rule.} Turning the FieldForensics map into a document score by its raw maximum gives AUC <<f3(R['aggregation']['max']['auc'])>>, the smoothed maximum of Eq.~(\ref{eq:score}) <<f3(R['aggregation']['smooth_max']['auc'])>>, the mean of the top-256 pixels <<f3(R['aggregation']['topk']['auc'])>>, and the global mean only <<f3(R['aggregation']['mean']['auc'])>>. Peak-based pooling is essential: averaging dilutes a sub-1\% region with genuine pixels, which is the same failure mode as whole-page classification.""")))

    # ------------------------------------------------------------ robustness
    rb = R.get("robust", {})
    if PK in rb and len(rb[PK]) > 1:
        A(("subsec", "Robustness to post-processing"))
        A(("fig", FIG + "robustness.png", "Image-level AUC under test-time degradations applied to the whole page (both classes alike): JPEG re-compression at quality 90/75/50, down-scaling to 75\\%/50\\%, Gaussian blur ($\\sigma=1$), Gaussian noise ($\\sigma=5$), and a messaging-app style re-share (75\\% scale + JPEG 80). Darker cells mean higher AUC; hatched cells (n/e) were not evaluated because of the compute budget.", "fig:robust", True, 7.0))
        A(("p", X(ROB_NOTE())))
        A(("fig", FIG + "tradeoff.png", "The accuracy--robustness trade-off: clean AUC against AUC after JPEG-75 re-compression of the page. Degradation augmentation (arrow) moves the network from the most accurate to the most robust operating point; the grey diagonal would mean no loss at all.", "fig:tradeoff", False, 3.3))
        A(("p", X(r"""Fig.~\ref{fig:tradeoff} summarises this trade-off. No system lies close to the diagonal. Every detector loses evidence under re-compression, and choosing between FieldForensics and FieldForensics-R means choosing a point on this frontier.""")))
        if has("robust_source"):
            A(("fig", FIG + "robust_source.png", "Per-source AUC under degradations for FieldForensics (solid) and FieldForensics-R (dotted). WildReceipt and SROIE are distributed as JPEG files, CORD and XFUND as PNG.", "fig:robsrc", False, 3.3))

    # ------------------------------------------------------------ LOSO
    lo = R.get("loso", {})
    if lo:
        A(("subsec", "Generalisation to an unseen document source"))
        rows = [[SRCN[s], f3(lo[s]["in_auc"]), f"{lo[s]['out_auc']:.3f} {ci(lo[s]['out_ci'])}", pc(lo[s]["in_pair"]), pc(lo[s]["out_pair"]), f3(lo[s]["in_pixf1"]), f3(lo[s]["out_pixf1"])] for s in SRC if s in lo]
        A(("tab", "Leave-one-source-out (FieldForensics-R configuration): test documents of a source scored by the model trained on all sources (in) and by a model trained without that source (out)", "tab:loso",
           ["Held-out", "AUC in", "AUC out [95\\% CI]", "Pair in", "Pair out", "F1 in", "F1 out"], rows, False, [0.17, 0.11, 0.28, 0.11, 0.11, 0.11, 0.11]))
        A(("p", X(LOSO_NOTE())))

    # ------------------------------------------------------------ qualitative
    A(("subsec", "Qualitative results"))
    A(("fig", FIG + "qualitative.png", "FieldForensics tampering maps for one test document per source (context window around the field). For each document the forged version (left pair) and its authentic counterpart (right pair) are shown with the document score $s$. The network is evaluated on full pages; crops are for display only.", "fig:qual", True, 7.0))
    A(("p", X(QUAL_NOTE())))
    return B


# ---------------------------------------------------------------- narrative helpers
def SOURCE_NOTE():
    ps, pso = R["per_source_main"], R["per_source"]
    s = r"""Table~\ref{tab:source} breaks the results down by source. FieldForensics is strong on all four: <<andjoin([SRCN[k] + ' ' + f3(R['per_source_main'][k]['auc']) for k in SRC])>>. The peak-hit rate reaches <<pc(max(R['per_source_main'][k]['peak_hit'] for k in SRC))>>\% on the best source. XFUND forms remain the hardest case: their 2480$\times$3508 pages hold many crisp printed lines that can produce false peaks, and they make up only 14\% of the training documents. """
    if "main_s0" in pso and "forensic_gbdt" in pso:
        s += r"""The comparison between FieldForensics-R and the forensic GBDT is revealing. On CORD, FieldForensics-R is far ahead (<<f3(R['per_source']['main_s0']['cord'])>> vs. <<f3(R['per_source']['forensic_gbdt']['cord'])>>). On the JPEG-distributed WildReceipt and SROIE receipts the hand-crafted statistics are better (<<f3(R['per_source']['forensic_gbdt']['wildreceipt'])>> vs. <<f3(R['per_source']['main_s0']['wildreceipt'])>> and <<f3(R['per_source']['forensic_gbdt']['sroie'])>> vs. <<f3(R['per_source']['main_s0']['sroie'])>>), and FieldForensics-R collapses on XFUND (<<f3(R['per_source']['main_s0']['xfund'])>>). A field regenerated by GPT-Image-2 and pasted into a page that was JPEG-compressed before the edit lacks the page's 8$\times$8 block-compression history, and ELA and blockiness statistics are designed to detect exactly this inconsistency \cite{catnet,doctamper}. """
        if has("featimp"):
            s += r"""Grouped permutation importance confirms what the GBDT relies on: shuffling the ELA features across test images lowers its AUC by <<f3(R['featimp']['all']['ELA'])>>, against <<f3(R['featimp']['all']['Noise residual'])>> for the noise-residual features and <<f3(R['featimp']['all']['JPEG blockiness'])>> for blockiness. ELA measures how a region responds to one more JPEG compression, so it is a direct probe of compression history. Its importance even on CORD (<<f3(R['featimp']['cord']['ELA'])>>) suggests that the PNG-distributed CORD photographs also carry an earlier, camera-side JPEG history. """
        s += r"""The augmentation of FieldForensics-R re-compresses half of its training crops and so teaches the network to \emph{ignore} compression history. The fused FieldForensics-R + GBDT re-introduces that cue and is better than either component on every source. FieldForensics, trained without augmentation, learns both cues by itself: it beats the fusion on <<sum(R['per_source_main'][k]['auc'] > R['per_source']['fusion'][k] for k in SRC)>> of 4 sources, and adding the GBDT to it brings nothing. """
    if R.get("calib"):
        s += r"""Score levels still differ between document types (the median score of authentic pages is <<f3(R['calib']['xfund']['auth_med'])>> on XFUND but <<f3(R['calib']['cord']['auth_med'])>> on CORD), which is why the TPR at 5\% FPR (<<pc(m0['tpr_at_fpr5'])>>\%) is lower than the paired accuracy suggests: a single global threshold is imperfectly calibrated across document types."""
    return s


def SIZE_NOTE():
    sz = R["size"]
    s = r"""The AUC is <<f3(R['size']['0']['auc'])>>, <<f3(R['size']['1']['auc'])>>, <<f3(R['size']['2']['auc'])>> and <<f3(R['size']['3']['auc'])>> from the smallest to the largest quartile. The smallest quartile is dominated by XFUND (<<R['size']['0']['n_xfund']>> of its <<R['size']['0']['n']>> forged documents), because a form field is a tiny fraction of a large page. """
    if sz["0"].get("auc_noxf") is not None:
        s += r"""Excluding XFUND, the smallest quartile reaches <<f3(R['size']['0']['auc_noxf'])>> (dotted line), so any apparent effect of the relative edit size is largely a document-type effect. """
    s += r"""Pixel F1 increases with field size (from <<f3(R['size']['0']['pix_f1'])>> to <<f3(R['size']['3']['pix_f1'])>>), as expected for a localisation metric dominated by boundary errors."""
    return s


def ABL_NOTE(abl):
    lines = []
    if "rgb" in abl:
        s_ = R['signif']['rgb']
        lines.append(r"""\emph{Noise stream.} Removing the SRM residual input changes the AUC from <<f3(AUCR)>> to <<f3(M['rgb']['auc'])>> <<sig('rgb')>>.""" +
                     (r""" High-pass residuals expose the generated glyphs' missing scan noise more directly than RGB alone.""" if s_['diff'] > 0 and s_['p'] < 0.05 else
                      r""" At native resolution, an ImageNet encoder applied to RGB already captures most of the high-frequency evidence, so the residual stream contributes little at this training budget."""))
    if "unpaired" in abl:
        lines.append(r"""\emph{Paired hard negatives.} Sampling authentic crops uniformly instead of at the field location changes the AUC from <<f3(AUCR)>> to <<f3(M['unpaired']['auc'])>> <<sig('unpaired')>>""" +
                     (r""" and lowers paired accuracy to <<pc(M['unpaired']['pair_acc'])>>\% and pixel F1 to <<f3(M['unpaired'].get('pix_f1'))>>. A plausible explanation is that, without paired negatives, the network can partly rely on the content shortcut ``numeric fields are suspicious'', which also fires on authentic documents \cite{geirhos}.""" if R['signif']['unpaired']['diff'] > 0 else "."))
    if "weak" in abl:
        lines.append((r"""\emph{Pixel supervision.} With image labels only, the top-$k$ pooled network does not learn at all within the budget (AUC <<f3(M['weak']['auc'])>>, pixel F1 <<f3(M['weak'].get('pix_f1'))>>). Finding a field that covers less than 1\% of the page from a single bit per image is a hard multiple-instance problem \cite{mil}, and the masks shipped with the benchmark are what make the task learnable at this scale.""" if M['weak']['auc'] < 0.55 else
                      r"""\emph{Pixel supervision.} With image labels only, the top-$k$ pooled network reaches <<f3(M['weak']['auc'])>> AUC <<sig('weak')>> and its maps localise poorly (pixel F1 <<f3(M['weak'].get('pix_f1'))>>), which confirms the value of the masks shipped with the benchmark.""") if M['weak']['auc'] < AUCR - 0.03 else
                     r"""\emph{Pixel supervision.} With image labels only, the network reaches <<f3(M['weak']['auc'])>> AUC <<sig('weak')>>, and masks mainly improve localisation (pixel F1 <<f3(M['weak'].get('pix_f1'))>> vs. <<f3(mR.get('pix_f1'))>>).""")
    if "noaug" in M:
        lines.append(r"""\emph{Degradation augmentation.} Removing the augmentation is the largest single change: the AUC rises from <<f3(AUCR)>> to <<f3(M['noaug']['auc'])>> <<sig('noaug')>>. The resulting model is FieldForensics; its cost in robustness is analysed next.""")
    return " ".join(lines)


def ROB_NOTE():
    rb = R["robust"]
    P = PK
    s = r"""Fig.~\ref{fig:robust} reports AUC under degradations applied to the whole page, so authentic and forged images are processed identically. """
    pc_ = [c for c in rb.get(P, {}) if c != "none"]
    if pc_:
        s += r"""FieldForensics drops from <<f3(AUC)>> on clean pages to """ + andjoin(f"{rb[P][c]['auc']:.3f} under {COND_NAMES[c]}" for c in pc_) + ". "
    rc = [c for c in rb.get("main_s0", {}) if c != "none"]
    if rc:
        s += r"""FieldForensics-R starts lower (<<f3(AUCR)>>) and obtains """ + andjoin(f"{rb['main_s0'][c]['auc']:.3f} under {COND_NAMES[c]}" for c in rc) + ". "
    both = [c for c in pc_ if c in rb.get("main_s0", {})]
    if both:
        better_R = [c for c in both if rb["main_s0"][c]["auc"] > rb[P][c]["auc"] + 0.01]
        if better_R:
            s += r"""Degradation augmentation therefore buys robustness at the price of clean accuracy: FieldForensics-R is ahead under """ + andjoin(COND_NAMES[c] for c in better_R) + \
                 r""", whereas FieldForensics is far ahead on pristine files. Which configuration to deploy depends on the channel through which documents arrive: original uploads favour FieldForensics, re-shared or re-compressed images favour FieldForensics-R, and a re-compression detector could select between them. """
        else:
            s += r"""Remarkably, FieldForensics remains ahead of FieldForensics-R even under these degradations, so at our training budget the augmentation costs clean accuracy without buying robustness. """
    if "forensic_gbdt" in rb and len(rb["forensic_gbdt"]) > 1:
        g = rb["forensic_gbdt"]
        gc = [c for c in g if c != "none"]
        s += r"""The hand-crafted compression statistics are the most fragile: the forensic GBDT falls from """ + f"{g['none']['auc']:.3f} to " + andjoin(f"{g[c]['auc']:.3f} under {COND_NAMES[c]}" for c in gc) + \
             r""", i.e. to chance. Their clean-data strength comes from the compression-history inconsistency of the pasted patch, which any re-compression, rescaling or blurring of the page erases. """
        if "fusion" in rb and len(rb["fusion"]) > 1:
            fu = rb["fusion"]
            fc = [c for c in fu if c != "none" and c in rb.get("main_s0", {})]
            if fc:
                worse = [c for c in fc if fu[c]["auc"] < rb["main_s0"][c]["auc"]]
                s += r"""For the same reason the FieldForensics-R + GBDT fusion obtains """ + andjoin(f"{fu[c]['auc']:.3f} under {COND_NAMES[c]}" for c in fc) + \
                     (r""", below FieldForensics-R alone, so a compression branch must be gated by a re-compression check before it is trusted. """ if worse else ". ")
    if has("robust_source") and "jpeg75" in R["robust_source"]:
        rs = R["robust_source"]
        drops = [rs['none'][k] - rs['jpeg75'][k] for k in SRC]
        s += f"Per source (Fig.~\\ref{{fig:robsrc}}), JPEG 75 costs FieldForensics {min(drops):.2f}--{max(drops):.2f} AUC on every source (" + \
             ", ".join(f"{SRCN[k]} {rs['none'][k] - rs['jpeg75'][k]:.3f}" for k in SRC) + \
             r"""), including the PNG-distributed ones. This is consistent with a network whose fine-grained evidence, both generator traces and compression history, is largely overwritten by one more round of compression."""
    if "global_cnn" in rb and len(rb["global_cnn"]) > 1:
        s += r""" The global ResNet-50 is at chance under every condition."""
    return s


def LOSO_NOTE():
    lo = R["loso"]
    ss = [s for s in SRC if s in lo]
    s = r"""To probe generalisation, we retrain the network (augmented configuration) with one source removed from training and validation and test it on that source's test documents (Table~\ref{tab:loso}). """
    s += "; ".join(f"{SRCN[k]} goes from {lo[k]['in_auc']:.3f} in-domain to {lo[k]['out_auc']:.3f} when held out" for k in ss) + ". "
    drops = {k: lo[k]["in_auc"] - lo[k]["out_auc"] for k in ss}
    if max(drops.values()) > 0.05:
        s += r"""Generator traces learned on one document population thus transfer only partially, and diverse training sources matter."""
    else:
        s += r"""The held-out source loses little, which suggests that the learned traces are largely independent of layout, language and acquisition."""
    return s


def QUAL_NOTE():
    return r"""Fig.~\ref{fig:qual} illustrates typical behaviour. On forged pages the edited value produces a compact, high-confidence blob, while the authentic counterpart yields a flat, low map. The maps are blob-shaped rather than following individual strokes, which limits pixel F1 even when the field is found. On large XFUND forms the authentic page can also produce elevated responses along printed lines, which is the main source of false alarms. Because the network outputs a localisation, an analyst can check every alarm by inspecting the highlighted field instead of trusting an opaque page-level score."""


def DISCUSSION():
    B = []
    A = B.append
    A(("p", X(r"""\emph{Why whole-page classification fails.} With a median forged area of 0.76\%, an authentic page and its forgery share more than 99\% of their pixels, and in the paired protocol they share everything else as well: layout, source, language and resolution. A page-level descriptor that pools uniformly over the image, such as a resized CNN input or a pixel histogram, averages the evidence away. Every method that works here, FieldForensics as well as the block-wise forensic statistics, keeps the evidence local and at native resolution and pools by maxima. Paired hard negatives keep the network from substituting content priors for forensic evidence. We think this \emph{localize-to-detect} view is the right default for any local generative edit, not only for receipts.""")))
    A(("p", X(r"""\emph{What is being detected.} Two families of cues are available. The first is the generator's own signature: the decoding and up-sampling structure of the image model, which renders glyphs whose noise and stroke texture differ from scanned or photographed ink. The second is an \emph{interface} cue: the pasted patch has a different compression history, sharpness and noise level from the page around it. Our experiments separate the two. Block-wise compression statistics see only interface cues: they are competitive on JPEG-distributed receipts but fall to chance after any re-compression. FieldForensics-R, whose augmentation suppresses compression cues, is weaker on clean data and needs the GBDT to recover them. The clean-trained FieldForensics learns both families and gains nothing from the GBDT. Interface cues are strong but fragile, and a careful forger could suppress them by re-saving the whole page. A deployable system should therefore treat re-compression as a first-class condition, as we do here, and combine pixel evidence with non-pixel checks such as arithmetic consistency of totals, issuer databases and provenance metadata (e.g., C2PA).""")))
    A(("p", X(r"""\emph{Relation to the published zero-shot results.} TruFor and DocTamper were not trained on generative field edits, and their low AUCs \cite{aiforge2} measure transfer, not the learnability of the task. Our results show that the task \emph{is} learnable from about two thousand paired examples on a single consumer GPU. The flip side is that a supervised detector inherits the biases of its training generator, sources and file-handling pipeline. Generalisation to other generators (e.g., the Gemini and Ideogram forgeries of AIForge-Doc v1) remains the critical open question.""")))
    A(("p", X(r"""\emph{Ethical considerations.} The dataset contains real receipts and forms released for research under non-commercial licences, and our models detect forgeries rather than create them. Reporting where detectors fail could inform forgers, but the failure modes we identify (re-compression, large-format pages) are already common knowledge in image forensics, and making them explicit helps practitioners avoid over-trusting automated checks. A detector score should support, not replace, human review and procedural controls.""")))
    return B


def LIMITS():
    B = []
    A = B.append
    lim = [r"""\textbf{Fragility to re-sharing.} Both configurations lose much of their accuracy when the page is re-compressed or rescaled. The best clean detector is not the most robust one, and closing this gap, for example with longer training or curriculum augmentation, is the most pressing practical problem.""",
           r"""\textbf{Single generator.} All forgeries come from GPT-Image-2, so the detector may be specific to its artefacts. Cross-generator evaluation on AIForge-Doc v1 (Gemini 2.5 Flash Image, Ideogram v2) is the most important next step.""",
           r"""\textbf{Compute budget.} All networks were trained for a short, equal budget (1,500 iterations, about 35 minutes) on a 6\,GB laptop GPU that was thermally throttled during the experiments, and each variant was run with a single seed. Bootstrap intervals capture test-sample uncertainty but not training variance.""",
           r"""\textbf{Large-format forms and calibration.} XFUND remains the hardest source, and score levels differ between document types, so per-type thresholds are needed in practice.""",
           r"""\textbf{Simulated degradations and paste-back forgeries.} Degradations were simulated digitally; print--scan and camera re-capture are not covered. The benchmark pastes the regenerated field back into the original page, whereas a forger who regenerates or re-photographs the whole page would remove the interface cues.""",
           r"""\textbf{Benchmark scale.} 606 test documents give AUC intervals of roughly $\pm$0.01--0.03 overall and wider ones per source (SROIE has 60 test documents)."""]
    A(("list", [X(l) for l in lim]))
    A(("p", r"""Future work will train on mixtures of generators, add a JPEG-DCT stream as in CAT-Net \cite{catnet}, use OCR to restrict attention to text lines, combine pixel evidence with semantic checks (e.g., totals that do not add up), and test on physically printed and re-captured documents."""))
    return B


def CONCL():
    return [("p", X(r"""We revisited the detection of AI-generated field forgeries in receipts and forms on the AIForge-Doc v2 benchmark. After de-duplicating the data and adopting a strictly paired protocol, we showed that classifying down-sampled whole pages fails because the edit is too small, and that evidence must be gathered locally and at native resolution. FieldForensics localises tampering with an RGB--noise U-Net trained on field-centred crops with paired hard negatives, and scores a page by the peak of its map. It reaches an AUC of <<f3(AUC)>>, a paired 2AFC accuracy of <<pc(m0['pair_acc'])>>\% and a pixel F1 of <<f3(m0['pix_f1'])>>, where humans and published zero-shot detectors are at 0.50--0.60, and it points to the forged field. Our analysis shows that such a detector exploits both generator traces and compression-history inconsistencies. The latter make block-wise forensic statistics a strong baseline on pristine files, but they vanish under re-compression. Training for robustness to degradations trades clean accuracy for robustness, and re-shared documents remain the main open problem. Our splits, code and per-image predictions will be made available to support further work on this threat."""))]
