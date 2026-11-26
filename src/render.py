"""Render the paper (list of blocks) to (1) an IEEEtran LaTeX source and (2) a two-column IEEE-style PDF via ReportLab.
Block types: ('sec',t) ('subsec',t) ('p',latex) ('list',[items]) ('enum',[items]) ('eq',latex,label)
('fig',path,caption,label,wide,width_in) ('tab',caption,label,header,rows,wide,colspec)"""
import re, os, io
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table, TableStyle,
                                NextPageTemplate, KeepTogether, CondPageBreak)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- fonts (Times-like with Unicode)
FD = os.environ.get("FONT_DIR", "C:/Windows/Fonts/")  # needs times.ttf, timesbd.ttf, timesi.ttf, timesbi.ttf
pdfmetrics.registerFont(TTFont("TNR", FD + "times.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", FD + "timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TNR-I", FD + "timesi.ttf"))
pdfmetrics.registerFont(TTFont("TNR-BI", FD + "timesbi.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")
pdfmetrics.registerFont(TTFont("SYM", FD + "seguisym.ttf")) if os.path.exists(FD + "seguisym.ttf") else None

# ---------------------------------------------------------------- latex-subset -> reportlab markup
SYM = {r"\times": "×", r"\approx": "≈", r"\le": "≤", r"\ge": "≥", r"\leq": "≤", r"\geq": "≥", r"\pm": "±", r"\sigma": "σ",
       r"\alpha": "α", r"\beta": "β", r"\lambda": "λ", r"\mu": "μ", r"\rightarrow": "→", r"\to": "→", r"\cdot": "·",
       r"\sim": "~", r"\infty": "∞", r"\Delta": "Δ", r"\delta": "δ", r"\ldots": "…", r"\dots": "…", r"\in": "∈",
       r"\%": "%", r"\&": "&amp;", r"\,": " ", r"\;": " ", r"\!": "", r"\quad": "  ", r"\mathrm": "", r"\text": "",
       r"\left": "", r"\right": "", r"\top": "T", r"\sum": "Σ", r"\log": "log", r"\exp": "exp", r"\max": "max", r"\min": "min",
       r"\checkmark": "✓", r"\neq": "≠", r"\hat": "", r"\textsc": "", r"\_": "_", r"\#": "#", r"\textasciitilde": "~", r"\ell": "ℓ",
       r"\dagger": "†", r"\ddagger": "‡", r"\mathcal": "", r"\`e": "è", r"\'e": "é", r"\'y": "ý", r'\"o': "ö", r'\"u': "ü"}


class Doc:
    def __init__(self):
        self.cites, self.labels = {}, {}

    def cite(self, keys):
        out = []
        for k in keys.split(","):
            k = k.strip()
            if k not in self.cites:
                self.cites[k] = len(self.cites) + 1
            out.append(str(self.cites[k]))
        return "[" + ", ".join(out) + "]"


D = Doc()


def math_rl(m):
    s = m.replace("\\{", "\x01").replace("\\}", "\x02")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(mathrm|text|textrm|operatorname|mathit)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", r"<b>\1</b>", s)
    for k in sorted(SYM, key=len, reverse=True):
        s = s.replace(k, SYM[k])
    s = re.sub(r"\^\{([^{}]*)\}", r"<super>\1</super>", s)
    s = re.sub(r"\^(\S)", r"<super>\1</super>", s)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)
    s = s.replace("{", "").replace("}", "")
    # italicise single latin letters (variables)
    s = re.sub(r"(?<![\w<>/&;])([A-Za-z])(?![\w>])", r"<i>\1</i>", s)
    return s


def tex2rl(t):
    t = t.replace("\\&", "\x03")
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("&amp;", "&amp;")
    t = re.sub(r"\\cite\{([^}]*)\}", lambda m: D.cite(m.group(1)), t)
    t = re.sub(r"\\ref\{([^}]*)\}", lambda m: str(D.labels.get(m.group(1), "?")), t)
    t = re.sub(r"\$([^$]+)\$", lambda m: math_rl(m.group(1).replace("&lt;", "<").replace("&gt;", ">")).replace("<<", "&lt;<"), t)
    for _ in range(3):
        t = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", t)
        t = re.sub(r"\\(emph|textit)\{([^{}]*)\}", r"<i>\2</i>", t)
        t = re.sub(r"\\texttt\{([^{}]*)\}", r"<font face='Courier' size='7.5'>\1</font>", t)
        t = re.sub(r"\\url\{([^{}]*)\}", r"<font face='Courier' size='7'>\1</font>", t)
    t = t.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”").replace("~", " ")
    for k in sorted(SYM, key=len, reverse=True):
        t = t.replace(k, SYM[k])
    t = t.replace("{", "").replace("}", "")
    t = t.replace("\x01", "{").replace("\x02", "}").replace("\x03", "&amp;")
    for ch in "∈✓":
        t = t.replace(ch, f"<font face='SYM'>{ch}</font>")
    return t


# ---------------------------------------------------------------- PDF
W, H = letter
LM = RM = 0.62 * inch
TM, BM = 0.75 * inch, 0.8 * inch
GAP = 0.25 * inch
CW = (W - LM - RM - GAP) / 2
ST = dict(
    body=ParagraphStyle("b", fontName="TNR", fontSize=9.6, leading=11.4, alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=1.5),
    sec=ParagraphStyle("s", fontName="TNR", fontSize=9.6, leading=12, alignment=TA_CENTER, spaceBefore=7, spaceAfter=4),
    sub=ParagraphStyle("ss", fontName="TNR-I", fontSize=9.6, leading=12, spaceBefore=5, spaceAfter=2),
    cap=ParagraphStyle("c", fontName="TNR", fontSize=7.8, leading=9.2, alignment=TA_JUSTIFY),
    tcap=ParagraphStyle("tc", fontName="TNR", fontSize=7.8, leading=9.4, alignment=TA_CENTER),
    cell=ParagraphStyle("cell", fontName="TNR", fontSize=7.3, leading=8.4, alignment=TA_CENTER),
    celll=ParagraphStyle("celll", fontName="TNR", fontSize=7.3, leading=8.4, alignment=TA_LEFT),
    li=ParagraphStyle("li", fontName="TNR", fontSize=9.6, leading=11.4, alignment=TA_JUSTIFY, leftIndent=12, bulletIndent=3, spaceAfter=1),
    ref=ParagraphStyle("r", fontName="TNR", fontSize=7.6, leading=8.9, alignment=TA_JUSTIFY, leftIndent=14, firstLineIndent=-14, spaceAfter=1.2),
    abs=ParagraphStyle("a", fontName="TNR-B", fontSize=8.8, leading=10.4, alignment=TA_JUSTIFY, spaceAfter=3),
)
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV"]


def eq_image(latex, num):
    plt.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(3.2, 0.32))
    latex = re.sub(r"\\[Bb]ig[lr]?(?=[(\[)\]])", "", latex)
    fig.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=10, family="serif")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    s = min(1.0, (CW - 30) / img.drawWidth) * 0.24 * 1  # 300dpi -> pt scaling
    iw, ih = img.imageWidth * 72 / 300, img.imageHeight * 72 / 300
    img.drawWidth, img.drawHeight = iw, ih
    t = Table([[img, Paragraph(f"({num})", ST["body"])]], colWidths=[CW - 26, 26])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (0, 0), "CENTER"), ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def make_table(header, rows, width, colw=None, bold_rows=()):
    data = [[Paragraph(f"<b>{tex2rl(h)}</b>", ST["cell"]) for h in header]]
    for i, r in enumerate(rows):
        data.append([Paragraph(("<b>%s</b>" if i in bold_rows else "%s") % tex2rl(str(c)), ST["celll"] if j == 0 else ST["cell"]) for j, c in enumerate(r)])
    if colw is None:
        first = width * 0.34
        colw = [first] + [(width - first) / (len(header) - 1)] * (len(header) - 1)
    else:
        colw = [width * c for c in colw]
    t = Table(data, colWidths=colw, repeatRows=1)
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                           ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return t


def build_pdf(meta, blocks, refs, out):
    # pre-pass: numbering for labels
    nf = nt = ne = ns = 0
    for b in blocks:
        if b[0] == "fig":
            nf += 1; D.labels[b[3]] = nf
        elif b[0] == "tab":
            nt += 1; D.labels[b[2]] = ROMAN[nt - 1]
        elif b[0] == "eq":
            ne += 1; D.labels[b[2]] = ne
        elif b[0] == "sec":
            ns += 1
            if len(b) > 2: D.labels[b[2]] = ROMAN[ns - 1]
    wide_items = []
    story = []

    def first_page(c, doc):
        y = H - TM
        p = Paragraph(meta["title"], ParagraphStyle("t", fontName="TNR", fontSize=20, leading=23.5, alignment=TA_CENTER))
        w, h = p.wrap(W - LM - RM - 40, 200); p.drawOn(c, LM + 20, y - h); y -= h + 12
        n = len(meta["authors"]); per = 4 if n > 3 else n
        colw = (W - LM - RM) / per
        maxh = 0
        for i, a in enumerate(meta["authors"]):
            ap = Paragraph(a, ParagraphStyle("au", fontName="TNR", fontSize=9.2, leading=10.8, alignment=TA_CENTER))
            w, h = ap.wrap(colw - 6, 200); ap.drawOn(c, LM + (i % per) * colw + 3, y - h); maxh = max(maxh, h)
        page_footer(c, doc)

    def page_footer(c, doc):
        c.setFont("TNR", 7.5)
        c.drawCentredString(W / 2, BM / 2, str(doc.page))

    def wide_page(item):
        def fn(c, doc):
            flow, caph = item["flow"], item["cap"]
            w, h = flow.wrap(W - LM - RM, H)
            y = H - TM
            flow.drawOn(c, LM + (W - LM - RM - w) / 2, y - h)
            cw, ch = caph.wrap(W - LM - RM, 200)
            caph.drawOn(c, LM, y - h - 3 - ch)
            page_footer(c, doc)
        return fn

    title_h = meta.get("title_h", 2.3 * inch)
    frames_first = [Frame(LM, BM, CW, H - TM - BM - title_h, id="f1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                    Frame(LM + CW + GAP, BM, CW, H - TM - BM - title_h, id="f2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
    frames = [Frame(LM, BM, CW, H - TM - BM, id="c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
              Frame(LM + CW + GAP, BM, CW, H - TM - BM, id="c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
    templates = [PageTemplate("first", frames_first, onPage=first_page), PageTemplate("normal", frames, onPage=page_footer)]

    story.append(NextPageTemplate("normal"))
    story.append(Paragraph("<i>Abstract</i>—" + tex2rl(meta["abstract"]), ST["abs"]))
    story.append(Paragraph("<i>Index Terms</i>—" + tex2rl(meta["keywords"]), ST["abs"]))
    sec_i = 0; sub_i = 0; eq_i = 0; fig_i = 0; tab_i = 0
    for b in blocks:
        k = b[0]
        if k == "sec":
            sec_i += 1; sub_i = 0
            title = b[1]
            txt = f"{ROMAN[sec_i - 1]}. {title.upper()}" if not b[1].startswith("*") else title[1:].upper()
            story.append(CondPageBreak(0.9 * inch))
            story.append(Paragraph(re.sub(r"(\b[A-Z])([A-Z]+)", lambda m: m.group(1) + f"<font size='7.6'>{m.group(2)}</font>", txt), ST["sec"]))
        elif k == "subsec":
            sub_i += 1
            story.append(CondPageBreak(0.6 * inch))
            story.append(Paragraph(f"{chr(64 + sub_i)}. {tex2rl(b[1])}", ST["sub"]))
        elif k == "p":
            story.append(Paragraph(tex2rl(b[1]), ST["body"]))
        elif k in ("list", "enum"):
            for i, it in enumerate(b[1]):
                story.append(Paragraph(tex2rl(it), ST["li"], bulletText="•" if k == "list" else f"{i + 1})"))
        elif k == "eq":
            eq_i += 1
            story.append(eq_image(b[1], eq_i))
        elif k == "fig":
            fig_i += 1
            _, path, cap, label, wide, wid = b
            cap_p = Paragraph(f"Fig. {fig_i}. " + tex2rl(cap), ST["cap"])
            img = Image(path)
            if wide:
                ww = wid * inch
                img.drawHeight = img.drawHeight * ww / img.drawWidth; img.drawWidth = ww
                item = {"flow": img, "cap": cap_p}
                cw_, ch_ = cap_p.wrap(W - LM - RM, 200)
                top_h = img.drawHeight + ch_ + 16
                name = f"wide{fig_i}"
                fr = [Frame(LM, BM, CW, H - TM - BM - top_h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                      Frame(LM + CW + GAP, BM, CW, H - TM - BM - top_h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
                templates.append(PageTemplate(name, fr, onPage=wide_page(item)))
                wide_items.append(name)
                story.append(NextPageTemplate(name))
            else:
                ww = min(CW, wid * inch)
                img.drawHeight = img.drawHeight * ww / img.drawWidth; img.drawWidth = ww
                story.append(KeepTogether([Spacer(1, 4), img, Spacer(1, 3), cap_p, Spacer(1, 6)]))
        elif k == "tab":
            tab_i += 1
            _, cap, label, header, rows, wide, colw = b[:7]
            bold = b[7] if len(b) > 7 else ()
            cap_p = Paragraph(f"TABLE {ROMAN[tab_i - 1]}<br/><font size='7.4'>{tex2rl(cap).upper()}</font>", ST["tcap"])
            if wide:
                t = make_table(header, rows, W - LM - RM, colw, bold)
                from reportlab.platypus.flowables import Flowable

                class Stack(Flowable):
                    def __init__(s, a, bb):
                        Flowable.__init__(s); s.a, s.b = a, bb

                    def wrap(s, aw, ah):
                        w1, h1 = s.a.wrap(aw, ah); w2, h2 = s.b.wrap(aw, ah); s.h1 = h1; s.hh = h1 + h2 + 3; return aw, s.hh

                    def draw(s):
                        s.b.drawOn(s.canv, 0, 0); s.a.drawOn(s.canv, 0, s.hh - s.h1)

                stack = Stack(cap_p, t)
                item = {"flow": stack, "cap": Paragraph("", ST["cap"])}
                _, sh = stack.wrap(W - LM - RM, H)
                name = f"widet{tab_i}"
                fr = [Frame(LM, BM, CW, H - TM - BM - sh - 14, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                      Frame(LM + CW + GAP, BM, CW, H - TM - BM - sh - 14, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
                templates.append(PageTemplate(name, fr, onPage=wide_page(item)))
                story.append(NextPageTemplate(name))
            else:
                t = make_table(header, rows, CW, colw, bold)
                story.append(KeepTogether([Spacer(1, 4), cap_p, Spacer(1, 3), t, Spacer(1, 7)]))
    # references
    story.append(CondPageBreak(0.8 * inch))
    story.append(Paragraph("R<font size='7.6'>EFERENCES</font>", ST["sec"]))
    order = sorted(D.cites.items(), key=lambda x: x[1])
    for k, n in order:
        story.append(Paragraph(f"[{n}]&nbsp;&nbsp;" + tex2rl(refs[k]), ST["ref"]))
    missing = [k for k in D.cites if k not in refs]
    if missing:
        print("MISSING REFS", missing)
    class QDoc(BaseDocTemplate):
        """Queue of full-width floats: each is placed at the top of the next free page."""
        wq = []

        def handle_nextPageTemplate(self, pt):
            if isinstance(pt, str) and pt.startswith("wide"):
                self.wq.append(pt)
                if len(self.wq) == 1:
                    BaseDocTemplate.handle_nextPageTemplate(self, pt)
            else:
                BaseDocTemplate.handle_nextPageTemplate(self, pt)

        def handle_pageBegin(self):
            BaseDocTemplate.handle_pageBegin(self)
            if self.wq and self.pageTemplate.id == self.wq[0]:
                self.wq.pop(0)
                BaseDocTemplate.handle_nextPageTemplate(self, self.wq[0] if self.wq else "normal")

    doc = QDoc(out, pagesize=letter, leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
                          title=re.sub("<[^>]+>", "", meta["title"]), author=meta.get("pdf_author", ""))
    doc.addPageTemplates(templates)
    doc.build(story)
    return order


# ---------------------------------------------------------------- LaTeX
def build_tex(meta, blocks, refs, out, order):
    L = [r"\documentclass[conference]{IEEEtran}", r"\IEEEoverridecommandlockouts",
         r"\usepackage{cite}", r"\usepackage{amsmath,amssymb,amsfonts}", r"\usepackage{graphicx}", r"\usepackage{booktabs}",
         r"\usepackage{textcomp}", r"\usepackage{xcolor}", r"\usepackage{url}", r"\usepackage[utf8]{inputenc}", r"\usepackage{newunicodechar}",
         r"\newunicodechar{×}{\ensuremath{\times}}", r"\newunicodechar{≈}{\ensuremath{\approx}}", r"\newunicodechar{σ}{\ensuremath{\sigma}}",
         r"\newunicodechar{–}{--}", r"\newunicodechar{—}{---}", r"\newunicodechar{±}{\ensuremath{\pm}}", r"\newunicodechar{→}{\ensuremath{\rightarrow}}",
         r"\newunicodechar{≥}{\ensuremath{\geq}}", r"\newunicodechar{Δ}{\ensuremath{\Delta}}", r"\newunicodechar{†}{\ensuremath{\dagger}}",
         r"\newunicodechar{“}{``}", r"\newunicodechar{”}{''}", r"\newunicodechar{’}{'}", r"\newunicodechar{≤}{\ensuremath{\leq}}", r"\newunicodechar{✓}{\ensuremath{\checkmark}}",
         r"\begin{document}", r"\title{" + re.sub("<[^>]+>", "", meta["title"]) + "}", r"\author{"]
    L.append("\n\\and\n".join(meta["authors_tex"]) + "}")
    L += [r"\maketitle", r"\begin{abstract}", meta["abstract"], r"\end{abstract}", r"\begin{IEEEkeywords}", meta["keywords"], r"\end{IEEEkeywords}"]
    for b in blocks:
        k = b[0]
        if k == "sec":
            if b[1].startswith("*"):
                L.append(r"\section*{" + b[1][1:] + "}")
            else:
                L.append(r"\section{" + b[1] + "}" + (r"\label{" + b[2] + "}" if len(b) > 2 else ""))
        elif k == "subsec":
            L.append(r"\subsection{" + b[1] + "}")
        elif k == "p":
            L.append(b[1] + "\n")
        elif k in ("list", "enum"):
            env = "itemize" if k == "list" else "enumerate"
            L.append(r"\begin{" + env + "}"); L += [r"\item " + i for i in b[1]]; L.append(r"\end{" + env + "}")
        elif k == "eq":
            L += [r"\begin{equation}", b[1], r"\label{" + b[2] + "}", r"\end{equation}"]
        elif k == "fig":
            _, path, cap, label, wide, wid = b
            env = "figure*" if wide else "figure"
            L += [r"\begin{" + env + "}[t]", r"\centering", r"\includegraphics[width=" + (r"\textwidth" if wide else r"\columnwidth") + "]{figures/" + os.path.basename(path) + "}",
                  r"\caption{" + cap + "}", r"\label{" + label + "}", r"\end{" + env + "}"]
        elif k == "tab":
            _, cap, label, header, rows, wide, colw = b[:7]
            env = "table*" if wide else "table"
            L += [r"\begin{" + env + "}[t]", r"\centering", r"\caption{" + cap + "}", r"\label{" + label + "}", r"\footnotesize",
                  r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
                  " & ".join(header) + r" \\", r"\midrule"]
            L += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
            L += [r"\bottomrule", r"\end{tabular}", r"\end{" + env + "}"]
    L.append(r"\begin{thebibliography}{99}")
    for k, n in order:
        L.append(r"\bibitem{" + k + "} " + refs[k])
    L += [r"\end{thebibliography}", r"\end{document}"]
    s = "\n".join(L)
    open(out, "w", encoding="utf8").write(s)
