from paths import ROOT, DATA  # noqa: E402
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render
from paper_content import build
from paper_refs import REFS
meta, blocks = build()
order = render.build_pdf(meta, blocks, REFS, ROOT + "paper/FieldForensics_paper.pdf")
render.build_tex(meta, blocks, REFS, ROOT + "paper/FieldForensics_paper.tex", order)
print("built", len(blocks), "blocks", len(order), "refs")
