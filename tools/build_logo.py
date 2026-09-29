#!/usr/bin/env python3
"""Build the CCVFM logo files from tools/logo/*.frag.svg.

All text is converted to vector outlines with fontTools, so the SVGs render the
same everywhere (GitHub does not load web fonts inside SVG images).

    python tools/build_logo.py --font-dir <dir with Unbounded-VF.ttf, IBMPlexMono-{Regular,Medium}.ttf>

Writes assets/logo_light.svg, assets/logo_dark.svg, assets/logo_icon.svg and
assets/social_preview.png (1280x640, for the repository's social preview).
Fonts (SIL OFL): github.com/google/fonts, ofl/unbounded and ofl/ibmplexmono.
"""
import argparse
import os

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S1, S2, S3 = "#eb6834", "#2a78d6", "#7a6fd0"          # Stage I / II / III
THEMES = {  # matched to GitHub's light and dark page backgrounds
    "light": dict(bg="#ffffff", data="#3d3d3a", muted="#59636e", ink="#1f2328", noise="#2a78d6",
                  rule="#d1d9e0"),
    "dark": dict(bg="#0d1117", data="#e6edf3", muted="#9198a1", ink="#e6edf3", noise="#86b6ef",
                 rule="#3d444d"),
}


class Face:
    def __init__(self, path, wght=None):
        f = TTFont(path)
        if wght is not None and "fvar" in f:
            f = instancer.instantiateVariableFont(f, {"wght": wght})
        self.font = f
        self.glyphs = f.getGlyphSet()
        self.cmap = f.getBestCmap()
        self.hmtx = f["hmtx"]
        self.upm = f["head"].unitsPerEm
        self.cap = f["OS/2"].sCapHeight / self.upm
        self.xh = f["OS/2"].sxHeight / self.upm

    def width(self, text, size, spacing=0.0):
        adv = sum(self.hmtx[self.cmap[ord(c)]][0] for c in text) * size / self.upm
        return adv + spacing * (len(text) - 1)

    def paths(self, text, size, x, y, spacing=0.0):
        """One SVG path string per glyph; (x, y) is the left end of the baseline."""
        s = size / self.upm
        out = []
        for c in text:
            g = self.cmap[ord(c)]
            pen = SVGPathPen(self.glyphs)
            self.glyphs[g].draw(TransformPen(pen, (s, 0, 0, -s, x, y)))
            d = pen.getCommands()
            if d:
                out.append(d)
            x += self.hmtx[g][0] * s + spacing
        return out


def path_el(ds, fill):
    return "".join(f'<path d="{d}" fill="{fill}"/>' for d in ds)


def runs_el(runs, size, x, y):
    """runs: [(face, text, fill)] laid out left to right."""
    out = []
    for face, text, fill in runs:
        out.append(path_el(face.paths(text, size, x, y), fill))
        x += face.width(text, size)
    return "".join(out)


def runs_width(runs, size):
    return sum(face.width(text, size) for face, text, _ in runs)


def fill_theme(frag, t):
    for k in ("bg", "data", "muted", "ink", "noise"):
        frag = frag.replace(f"@{k.upper()}@", t[k])
    return frag


def build(theme, fonts, stages_frag, mark_frag):
    t = THEMES[theme]
    word, mono, mono_med = fonts
    pad = 12
    # ---- three-stage strip (native 540 x 140 units) ----
    sw = 720.0
    sh = sw * 140 / 540
    labels = []
    for cx, num, col, text in [(84, "I", S1, " coreset"), (270, "II", S2, " closed-form velocity law"),
                               (446, "III", S3, " learned correction")]:
        runs = [(mono_med, num, col), (mono, text, t["muted"])]
        labels.append(runs_el(runs, 9, cx - runs_width(runs, 9) / 2, 134))
    # ---- lockup: mark / wordmark / full name ----
    ws, wsp = 72.0, 3.0
    letters = [("C", S1), ("C", S2), ("V", S2), ("F", S3), ("M", S3)]
    word_w = word.width("CCVFM", ws, wsp)
    ts = 13.0
    tag = [(mono, "coreset", S1), (mono, "-induced ", t["muted"]), (mono, "conditional velocity", S2),
           (mono, " ", t["muted"]), (mono, "flow matching", S3)]
    tag_w = runs_width(tag, ts)
    mark_w, mark_h = 150.0, 112.5
    lock_w = max(mark_w, word_w, tag_w)
    lock_h = mark_h + 18 + ws * word.cap + 22 + ts * mono.cap
    H = max(sh, lock_h) + 2 * pad
    sep_x = sw + 30
    lx = sep_x + 30
    W = lx + lock_w + pad
    top = (H - lock_h) / 2
    cx = lx + lock_w / 2
    word_base = top + mark_h + 18 + ws * word.cap
    x = cx - word_w / 2
    word_paths = []
    for ch, col in letters:
        word_paths.append(path_el(word.paths(ch, ws, x, word_base), col))
        x += word.width(ch, ws) + wsp
    tag_base = word_base + 22 + ts * mono.cap
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.1f} {H:.1f}" '
        f'width="{W:.0f}" height="{H:.0f}" role="img" aria-label="CCVFM">'
        f'<title>CCVFM: Coreset-Induced Conditional Velocity Flow Matching</title>'
        f'<svg x="0" y="{(H - sh) / 2:.1f}" width="{sw:.1f}" height="{sh:.1f}" viewBox="0 0 540 140">'
        f'{fill_theme(stages_frag, t)}{"".join(labels)}</svg>'
        f'<line x1="{sep_x:.1f}" y1="{H / 2 - 88:.1f}" x2="{sep_x:.1f}" y2="{H / 2 + 88:.1f}" '
        f'stroke="{t["rule"]}" stroke-width="1"/>'
        f'<svg x="{cx - mark_w / 2:.1f}" y="{top:.1f}" width="{mark_w}" height="{mark_h}" '
        f'viewBox="0 0 160 120">{fill_theme(mark_frag, t).replace("streamW", "streamW_" + theme)}</svg>'
        f'{"".join(word_paths)}{runs_el(tag, ts, cx - tag_w / 2, tag_base)}</svg>\n')


def build_icon(mark_frag):
    t = THEMES["light"]
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" '
            'role="img" aria-label="CCVFM"><rect width="512" height="512" rx="96" fill="#ffffff"/>'
            f'<svg x="56" y="86" width="400" height="300" viewBox="0 0 160 120">'
            f'{fill_theme(mark_frag, t).replace("streamW", "streamW_icon")}</svg></svg>\n')


def social_preview(font_dir, word_ttf, word, mono):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    wp = FontProperties(fname=word_ttf)
    mp = FontProperties(fname=os.path.join(font_dir, "IBMPlexMono-Regular.ttf"))
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(640, 0)
    ax.axis("off")
    px = lambda pt: pt * 100 / 72  # noqa: E731  (points -> pixels at dpi 100)
    letters = [("C", S1), ("C", S2), ("V", S2), ("F", S3), ("M", S3)]
    sp = 4.0
    x = 640 - word.width("CCVFM", px(64), sp) / 2
    for ch, col in letters:
        ax.text(x, 100, ch, fontproperties=wp, fontsize=64, color=col, va="baseline")
        x += word.width(ch, px(64)) + sp
    parts = [("coreset", S1), ("-induced ", "#59636e"), ("conditional velocity", S2), (" ", "#59636e"),
             ("flow matching", S3)]
    x = 640 - sum(mono.width(s, px(17)) for s, _ in parts) / 2
    for text, col in parts:
        ax.text(x, 142, text, fontproperties=mp, fontsize=17, color=col, va="baseline")
        x += mono.width(text, px(17))
    ax.text(640, 178, "NeurIPS 2026  ·  github.com/ziwaa-se/CCVFM", fontproperties=mp, fontsize=15,
            color="#59636e", ha="center", va="baseline")
    img = mpimg.imread(os.path.join(ROOT, "assets", "method.png"))
    h, w = img.shape[:2]
    y0, y1 = 205, 625
    bw = min(1200.0, (y1 - y0) * w / h)
    bh = bw * h / w
    ax.imshow(img, extent=(640 - bw / 2, 640 + bw / 2, y0 + bh, y0), aspect="auto")
    ax.set_xlim(0, 1280)
    ax.set_ylim(640, 0)
    out = os.path.join(ROOT, "assets", "social_preview.png")
    fig.savefig(out, dpi=100, facecolor="white")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--font-dir", required=True)
    args = p.parse_args()
    word = Face(os.path.join(args.font_dir, "Unbounded-VF.ttf"), wght=700)
    word_ttf = os.path.join(args.font_dir, "Unbounded-Bold-static.ttf")
    word.font.save(word_ttf)
    mono = Face(os.path.join(args.font_dir, "IBMPlexMono-Regular.ttf"))
    mono_med = Face(os.path.join(args.font_dir, "IBMPlexMono-Medium.ttf"))
    frag = lambda n: open(os.path.join(ROOT, "tools", "logo", n)).read()  # noqa: E731
    stages, mark = frag("stages.frag.svg"), frag("mark.frag.svg")
    for theme in THEMES:
        out = os.path.join(ROOT, "assets", f"logo_{theme}.svg")
        open(out, "w").write(build(theme, (word, mono, mono_med), stages, mark))
        print("wrote", out)
    open(os.path.join(ROOT, "assets", "logo_icon.svg"), "w").write(build_icon(mark))
    print("wrote", os.path.join(ROOT, "assets", "logo_icon.svg"))
    social_preview(args.font_dir, word_ttf, word, mono)


if __name__ == "__main__":
    main()
