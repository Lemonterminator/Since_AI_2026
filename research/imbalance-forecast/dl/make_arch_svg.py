"""Generate the DL architecture diagram (SVG) with LaTeX formulas rendered by matplotlib mathtext."""
import io
import re
import sys
from html import escape

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["svg.hashsalt"] = "tt-dl"
plt.rcParams["mathtext.fontset"] = "cm"

OUT = sys.argv[1]
W, H = 1900, 1830
FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC','Segoe UI',sans-serif"
_fid = [0]
parts: list[str] = []


def tex(expr: str, x: float, y: float, size: float = 15, anchor: str = "start", color: str = "#1f2933") -> tuple[float, float]:
    """Render a mathtext formula and embed it as a nested <svg>. (x, y) = left/center/right of the vertical middle."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, expr, fontsize=size, color=color)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index("<svg"):]
    m = re.search(r'width="([\d.]+)pt" height="([\d.]+)pt" viewBox="([^"]+)"', s)
    w_pt, h_pt, vb = float(m.group(1)), float(m.group(2)), m.group(3)
    _fid[0] += 1
    p = f"f{_fid[0]}_"
    s = re.sub(r'id="([^"]+)"', lambda mm: f'id="{p}{mm.group(1)}"', s)
    s = re.sub(r'href="#([^"]+)"', lambda mm: f'href="#{p}{mm.group(1)}"', s)
    s = re.sub(r"url\(#([^)]+)\)", lambda mm: f"url(#{p}{mm.group(1)})", s)
    k = 1.333  # pt -> px
    w, h = w_pt * k, h_pt * k
    x0 = x if anchor == "start" else (x - w / 2 if anchor == "middle" else x - w)
    inner = s[s.index(">") + 1: s.rindex("</svg>")]
    parts.append(f'<svg x="{x0:.1f}" y="{y - h / 2:.1f}" width="{w:.1f}" height="{h:.1f}" viewBox="{vb}" overflow="visible">{inner}</svg>')
    return w, h


STYLE = {
    "data": ("#f1f3f5", "#868e96", ""),
    "feat": ("#e7f5ff", "#4dabf7", ""),
    "op": ("#e6fcf5", "#20c997", ""),
    "cell": ("#f3f0ff", "#7950f2", "6 4"),
    "q": ("#fff4e6", "#fd7e14", ""),
    "dir": ("#e7f5ff", "#228be6", ""),
    "spk": ("#fff0f6", "#e64980", ""),
    "loss": ("#fff9db", "#f08c00", ""),
    "label": ("#ebfbee", "#40c057", ""),
    "lgb": ("#f8f9fa", "#495057", "3 3"),
    "panel": ("#ffffff", "#ced4da", ""),
}


def box(x, y, w, h, kind, lines=(), title=None, size=13, rx=8, title_size=14, bold_first=False):
    fill, stroke, dash = STYLE[kind]
    da = f' stroke-dasharray="{dash}"' if dash else ""
    sw = 2 if kind == "cell" else 1.4
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}/>')
    items = []
    if title:
        items.append((title, title_size, "700", "#1f2933"))
    for i, ln in enumerate(lines):
        wt = "700" if (bold_first and i == 0 and not title) else "400"
        items.append((ln, size, wt, "#343a40"))
    if not items:
        return
    lh = [s * 1.38 for _, s, _, _ in items]
    total = sum(lh)
    cy = y + h / 2 - total / 2
    for (t, s, wt, col), l in zip(items, lh):
        cy += l
        parts.append(f'<text x="{x + w / 2}" y="{cy - l * 0.3:.1f}" font-size="{s}" font-weight="{wt}" fill="{col}" text-anchor="middle">{escape(t)}</text>')


def text(x, y, t, size=13, anchor="start", weight="400", color="#343a40", italic=False):
    st = ' font-style="italic"' if italic else ""
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{st}>{escape(t)}</text>')


def arrow(x1, y1, x2, y2, label=None, dashed=False, color="#495057", lx=0, ly=-6, curve=None, size=12):
    da = ' stroke-dasharray="5 4"' if dashed else ""
    if curve:
        cx, cy = curve
        d = f"M{x1},{y1} Q{cx},{cy} {x2},{y2}"
    else:
        d = f"M{x1},{y1} L{x2},{y2}"
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{da} marker-end="url(#ah)"/>')
    if label:
        mx, my = ((x1 + x2) / 2, (y1 + y2) / 2) if not curve else (curve[0], curve[1])
        parts.append(f'<text x="{mx + lx}" y="{my + ly}" font-size="{size}" fill="#495057" text-anchor="middle">{escape(label)}</text>')


def poly_arrow(points, dashed=False, color="#495057", label=None, lpos=None):
    da = ' stroke-dasharray="5 4"' if dashed else ""
    d = "M" + " L".join(f"{a},{b}" for a, b in points)
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{da} marker-end="url(#ah)"/>')
    if label and lpos:
        text(lpos[0], lpos[1], label, 12, "middle", color="#495057")


def section(x, y, w, h, title):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="#ced4da" stroke-width="1.2"/>')
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="12" fill="#f8f9fa" stroke="none"/>')
    parts.append(f'<line x1="{x}" y1="{y + 34}" x2="{x + w}" y2="{y + 34}" stroke="#e9ecef"/>')
    text(x + 16, y + 23, title, 16, weight="700", color="#212529")


def card(x, y, w, h, title, sub, rows, kind_rows=None, note=None, head=None):
    """Mini DAG card: rows = list of rows, each row a list of node labels drawn side by side."""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fcfcfd" stroke="#7950f2" stroke-width="1.6" stroke-dasharray="6 4"/>')
    text(x + 12, y + 22, title, 14, weight="700", color="#5f3dc4")
    text(x + 12, y + 40, sub, 11.5, color="#6c757d")
    top = y + 52
    bottom = y + h - (40 if note else 12) - (30 if head else 0)
    n = len(rows)
    gap = 12
    rh = (bottom - top - gap * (n - 1)) / n
    rh = min(rh, 40)
    used = n * rh + gap * (n - 1)
    top += (bottom - top - used) / 2
    centers = []
    for i, row in enumerate(rows):
        ry = top + i * (rh + gap)
        k = len(row)
        nw = (w - 24 - 8 * (k - 1)) / k
        cs = []
        for j, lab in enumerate(row):
            nx = x + 12 + j * (nw + 8)
            kind = (kind_rows[i] if kind_rows else "op")
            if isinstance(lab, tuple):
                lab, kind = lab
            lines = lab.split("\n")
            box(nx, ry, nw, rh, kind, lines, size=11.5 if len(lines) == 1 else 10.5, rx=6)
            cs.append((nx + nw / 2, ry, ry + rh))
        centers.append(cs)
    for a, b in zip(centers, centers[1:]):
        if len(a) == len(b):
            for (ax, _, ab), (bx, bt, _) in zip(a, b):
                arrow(ax, ab, bx, bt - 2)
        else:
            for (ax, _, ab) in a:
                for (bx, bt, _) in b:
                    arrow(ax, ab, bx, bt - 2)
    yy = y + h - 12
    if head:
        hy = yy - (40 if note else 0) - 22
        box(x + 12, hy - 4, w - 24, 26, "cell", [head], size=11.5, rx=6)
        last = centers[-1]
        for (ax, _, ab) in last:
            arrow(ax, ab, x + w / 2 if len(last) == 1 else ax, hy - 6)
    if note:
        for i, ln in enumerate(note.split("\n")):
            text(x + w / 2, y + h - 26 + i * 15, ln, 11, "middle", color="#495057")


# ---------------------------------------------------------------------------
# Title + legend
# ---------------------------------------------------------------------------
text(30, 42, "Token Titans · 40-step probabilistic imbalance price forecast · deep learning architectures", 24, weight="700", color="#212529")
text(30, 66, "One sample = one forecast origin + the next 40 quarter-hours (10 hours); all architectures share features, labels and loss, and differ only in the Backbone cell.", 13.5, color="#495057")
legend = [("cell", "expandable cell (dashed, expanded below)", 1330, 22), ("feat", "feature / tensor", 1630, 22),
          ("op", "operation", 1330, 48), ("label", "label", 1450, 48), ("loss", "loss", 1630, 48)]
for k, t, xx, yy in legend:
    fill, stroke, dash = STYLE[k]
    da = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{xx}" y="{yy}" width="22" height="16" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{da}/>')
    text(xx + 28, yy + 13, t, 12)

# ---------------------------------------------------------------------------
# Section A: main DAG
# ---------------------------------------------------------------------------
section(20, 84, 1860, 600, "A · Main data flow (DAG): features → Encode cell → Backbone cell → Prob head cell → three output heads + loss")

box(40, 300, 170, 96, "data", ["panel row", "(origin, step = 1…40)", "18 features", "(+ regime_pre_june)"], size=12, bold_first=True)
box(250, 136, 260, 196, "feat", ["last_spread · last_direction", "up_share_8q · spread_mean_8q", "spread_std_8q · spread_change_1q", "last_mfrr_up · last_mfrr_down", "wind_err_last"], title="origin state ×9 (same for all 40 steps)", size=12)
box(250, 366, 260, 196, "feat", ["step · delivery_hour · delivery_weekday", "spot_level · spot_ramp_1h", "wind_da · net_load_da", "net_load_ramp_1h · load_da_minus_latest", "（+ regime_pre_june）"], title="per-step known ×9 (differ by step)", size=12)
arrow(210, 335, 248, 240)
arrow(210, 362, 248, 460)

box(548, 196, 150, 76, "cell", ["⊞ Encode cell", "see B1"], size=13, bold_first=True)
box(548, 426, 150, 76, "cell", ["⊞ Encode cell", "see B1"], size=13, bold_first=True)
arrow(510, 234, 546, 234)
arrow(510, 464, 546, 464)

box(735, 200, 120, 68, "feat", ["X_state", "[B, Ds]"], size=13, bold_first=True)
box(735, 430, 120, 68, "feat", ["X_step", "[B, 40, Dh]"], size=13, bold_first=True)
arrow(698, 234, 733, 234)
arrow(698, 464, 733, 464)

box(900, 128, 200, 52, "lgb", ["LightGBM out-of-sample forecast", "used only by #5 lgbm_tide_corrector"], size=11.5)
box(900, 270, 200, 160, "cell", ["⊞ Backbone cell", "one of 9", "see C"], size=14, bold_first=True)
arrow(1000, 180, 1000, 268, dashed=True)
arrow(855, 234, 898, 310)
arrow(855, 464, 898, 392)

box(1140, 316, 92, 68, "feat", ["h", "[B,40,d]"], size=13, bold_first=True)
arrow(1100, 350, 1138, 350)
box(1268, 270, 170, 160, "cell", ["⊞ Prob head cell", "#1–#4: hierarchical head B2", "#6–#9: two-tower head B3", "#5: own correction head C5"], size=12, bold_first=True)
arrow(1232, 350, 1266, 350)
# highway
poly_arrow([(795, 498), (795, 560), (1353, 560), (1353, 432)], dashed=True, label="raw_in linear highway (X_state ‖ X_step)", lpos=(1075, 552))

# three heads
hx, hw = 1470, 140
heads = [("q", 110, ["Quantile head", "19 quantiles", "q̂₀.₀₅ ≤ … ≤ q̂₀.₉₅"]),
         ("dir", 290, ["Direction head", "3-class softmax", "up / down / none"]),
         ("spk", 440, ["Spike head", "3-class softmax", "normal / high / low"])]
for kind, yy, lines in heads:
    box(hx, yy, hw, 92 if kind != "q" else 140, kind, lines, size=12, bold_first=True)
arrow(1438, 300, 1468, 180)
arrow(1438, 350, 1468, 336)
arrow(1438, 400, 1468, 486)

# formulas next to heads
fx = 1626
tex(r"$\mathcal{L}_q = 0.45\,\frac{1}{19}\sum_{\tau}\rho_\tau(y-\hat q_\tau)$", fx, 136, 12)
tex(r"$\quad +\,0.35\,|y-\hat q_{0.5}| + 0.20\,\frac{\rho_{0.1}+\rho_{0.9}}{2}$", fx, 178, 12)
tex(r"$\rho_\tau(u)=\max(\tau u,(\tau-1)u),\ \tau\in\{0.05,\dots,0.95\}$", fx, 222, 9)
tex(r"$\mathcal{L}_{dir} = 0.35\,\mathrm{CE} + 0.65\sum_{k}(p_k-\mathbf{1}[y=k])^2$", fx, 326, 10.5)
text(fx, 368, "cross-entropy + 3-class Brier", 11.5, color="#495057")
tex(r"$\mathcal{L}_{spk} = 0.35\,\mathrm{CE} + 1.5\,\frac{(p_{hi}-o_{hi})^2+(p_{lo}-o_{lo})^2}{2}$", fx, 474, 10.5)
text(fx, 518, "o = 1[spread > +50] / 1[spread < −50]", 11.5, color="#495057")
for kind, yy in (("q", 180), ("dir", 336), ("spk", 486)):
    arrow(1610, yy, 1622, yy)

# labels + total loss
box(40, 594, 700, 80, "label", [], size=12)
text(56, 616, "Labels y (one network, three tasks)", 13.5, weight="700", color="#2b8a3e")
tex(r"$y=\frac{p^{imb}-p^{spot}}{s},\ \ s=\max(\mathrm{IQR}_{train},10)$", 56, 648, 11)
tex(r"$m_{o,t}=\mathbf{1}[t^{pub}_{o,t}\leq \mathrm{cutoff}]$", 330, 648, 11)
text(530, 616, "direction 3 classes · spike ±50 €", 12, color="#343a40")
text(530, 664, "m = 0: label zeroed, not in loss", 12, color="#343a40")

box(1140, 594, 720, 80, "loss", [], size=12)
text(1156, 616, "Total loss (mean over (origin, step) cells with mask m = 1)", 13.5, weight="700", color="#e67700")
tex(r"$\mathcal{L} = \mathcal{L}_q + 0.35\,\mathcal{L}_{dir} + 0.45\,\mathcal{L}_{spk}$", 1156, 650, 13)
text(1560, 656, "weights set by hand", 12, color="#868e96", italic=True)
arrow(740, 634, 1138, 634, label="y, m", ly=-8)

# ---------------------------------------------------------------------------
# Section B: cell expansions
# ---------------------------------------------------------------------------
section(20, 704, 1860, 400, "B · Modules expanded")

# B1 Encode cell
bx, by = 40, 748
parts.append(f'<rect x="{bx}" y="{by}" width="560" height="340" rx="10" fill="#fcfcfd" stroke="#7950f2" stroke-width="1.6" stroke-dasharray="6 4"/>')
text(bx + 12, by + 22, "B1 · Encode cell (per feature; scaler fitted on training rows only)", 14, weight="700", color="#5f3dc4")
box(bx + 200, by + 40, 160, 40, "feat", ["raw feature x"], size=12)
box(bx + 200, by + 100, 160, 40, "op", ["NaN → training median"], size=12)
arrow(bx + 280, by + 80, bx + 280, by + 98)
box(bx + 150, by + 160, 260, 50, "op", [], size=12)
tex(r"$z=\frac{x-\mathrm{med}(x)}{\mathrm{IQR}(x)}$", bx + 280, by + 185, 11, anchor="middle")
arrow(bx + 280, by + 140, bx + 280, by + 158)
cols = [("clip(z, ±6) / 3", "bounded linear"), ("asinh(z) / 3", "heavy tail"), ("1[NaN]", "missing flag*"), ("sin / cos", "hour·weekday·step")]
cw = 122
for i, (a, b) in enumerate(cols):
    cx = bx + 20 + i * (cw + 10)
    box(cx, by + 236, cw, 48, "op", [a, b], size=11)
    arrow(bx + 280, by + 210, cx + cw / 2, by + 234)
box(bx + 150, by + 296, 260, 30, "feat", ["concat → Ds / Dh channels"], size=12)
for i in range(4):
    cx = bx + 20 + i * (cw + 10) + cw / 2
    arrow(cx, by + 284, bx + 280, by + 294)
text(bx + 548, by + 334, "* when missing or for selected features", 10.5, "end", color="#868e96")

# B2 Hierarchical head
bx = 620
parts.append(f'<rect x="{bx}" y="{by}" width="620" height="340" rx="10" fill="#fcfcfd" stroke="#7950f2" stroke-width="1.6" stroke-dasharray="6 4"/>')
text(bx + 12, by + 22, "B2 · Hierarchical non-crossing head (#1–#4)", 14, weight="700", color="#5f3dc4")
box(bx + 20, by + 44, 120, 36, "feat", ["h"], size=12)
box(bx + 20, by + 96, 120, 36, "op", ["LayerNorm"], size=12)
arrow(bx + 80, by + 80, bx + 80, by + 94)
box(bx + 170, by + 44, 120, 36, "feat", ["raw_in"], size=12)
box(bx + 20, by + 150, 270, 36, "op", ["Linear(19) + highway Linear(19)  ⊕"], size=11.5)
arrow(bx + 80, by + 132, bx + 100, by + 148)
arrow(bx + 230, by + 80, bx + 230, by + 148)
box(bx + 20, by + 204, 270, 36, "feat", ["r ∈ R¹⁹"], size=12)
arrow(bx + 155, by + 186, bx + 155, by + 202)
br = [("r₀…r₈", "↓ lower cumsum"), ("r₉", "median"), ("r₁₀…r₁₈", "↑ upper cumsum")]
for i, (a, b) in enumerate(br):
    cx = bx + 20 + i * 92
    box(cx, by + 258, 86, 48, "op", [a, b], size=11)
    arrow(bx + 155, by + 240, cx + 43, by + 256)
box(bx + 20, by + 316, 270, 18, "q", ["19 q̂, monotone"], size=11)
tex(r"$\hat q_{0.5}=r_9$", bx + 320, by + 210, 11)
tex(r"$\hat q_{0.5+0.05k}=\hat q_{0.5}+\sum_{j\leq k}\left(\mathrm{softplus}(r_{9+j})+\epsilon\right)$", bx + 320, by + 246, 10)
tex(r"$\hat q_{0.5-0.05k}=\hat q_{0.5}-\sum_{j\leq k}\left(\mathrm{softplus}(r_{9-j})+\epsilon\right)$", bx + 320, by + 282, 10)
text(bx + 320, by + 66, "• biases start at training quantiles", 12)
text(bx + 320, by + 88, "• direction / spike: Linear(3) + highway", 12)
text(bx + 320, by + 110, "  bias = log(training class frequency)", 12)
text(bx + 320, by + 132, "• AQCR = 0 by construction (no crossing)", 12)
text(bx + 320, by + 154, "• seed average stays monotone (convex)", 12)

# B3 Decoupled dual head
bx = 1260
parts.append(f'<rect x="{bx}" y="{by}" width="600" height="340" rx="10" fill="#fcfcfd" stroke="#7950f2" stroke-width="1.6" stroke-dasharray="6 4"/>')
text(bx + 12, by + 22, "B3 · Decoupled two-tower head (#6–#9): median gets L1 gradient only", 14, weight="700", color="#5f3dc4")
box(bx + 30, by + 46, 150, 36, "feat", ["h_med (median tower)"], size=12)
box(bx + 330, by + 46, 150, 36, "feat", ["h_dist (distribution tower)"], size=12)
box(bx + 30, by + 106, 150, 40, "op", ["LN → Linear(1)", "+ highway"], size=11)
arrow(bx + 105, by + 82, bx + 105, by + 104)
box(bx + 30, by + 170, 150, 36, "q", ["q̂₀.₅"], size=12)
arrow(bx + 105, by + 146, bx + 105, by + 168)
box(bx + 200, by + 170, 110, 36, "op", ["stop-grad sg(·)"], size=11)
arrow(bx + 180, by + 188, bx + 198, by + 188)
box(bx + 330, by + 106, 250, 40, "op", ["LN → 9 upper · 9 lower gaps", "softplus + ε"], size=11)
arrow(bx + 405, by + 82, bx + 405, by + 104)
box(bx + 330, by + 170, 250, 36, "op", ["sg(q̂₀.₅) ± cumsum(gaps)"], size=11.5)
arrow(bx + 405, by + 146, bx + 405, by + 168)
arrow(bx + 310, by + 188, bx + 328, by + 188)
box(bx + 370, by + 226, 100, 36, "dir", ["dir. logits"], size=11.5)
box(bx + 480, by + 226, 100, 36, "spk", ["spike logits"], size=11.5)
poly_arrow([(bx + 580, by + 126), (bx + 592, by + 126), (bx + 592, by + 216), (bx + 530, by + 216), (bx + 530, by + 224)])
poly_arrow([(bx + 592, by + 216), (bx + 420, by + 216), (bx + 420, by + 224)])
box(bx + 30, by + 282, 550, 30, "q", ["concat → 19 quantiles (monotone)"], size=12)
arrow(bx + 105, by + 206, bx + 105, by + 280)
arrow(bx + 345, by + 206, bx + 345, by + 280)
tex(r"$\hat q_{\tau\neq 0.5}=\mathrm{sg}(\hat q_{0.5})\pm\sum\mathrm{softplus}(\cdot)\ \Rightarrow\ \partial\rho_{\tau\neq0.5}/\partial\hat q_{0.5}=0$", bx + 300, by + 328, 9.5, anchor="middle")

# ---------------------------------------------------------------------------
# Section C: backbone gallery
# ---------------------------------------------------------------------------
section(20, 1124, 1860, 520, "C · Backbone cell expanded (1 of 9; input X_state, X_step, output h)")
cw, ch = 358, 226
xs = [40 + i * (cw + 10) for i in range(5)]
y1, y2 = 1168, 1406
HEAD1 = "→ hierarchical head B2"
HEAD2 = "→ two-tower head B3"
card(xs[0], y1, cw, ch, "#1 tabular_resnet", "Per-step MLP, the same view as GBDT", [
    ["X_state repeated 40 steps ‖ X_step"], ["Fourier feature embedding (6 freq.)"], ["GatedResBlock ×3（128）"]],
    kind_rows=["feat", "op", "op"], head=HEAD1)
card(xs[1], y1, cw, ch, "#2 tide", "TiDE encoder-decoder (all 40 steps)", [
    ["cov_proj: X_step → 40×12"], ["‖ X_state → dense encoder GRB×2 (192)"], ["dense decoder → 40×48"],
    ["temporal decoder GRB (‖ covariates·state·step)"]], kind_rows=["op", "op", "op", "op"], head=HEAD1)
card(xs[2], y1, cw, ch, "#3 conv_bigru", "Multi-scale conv + FiLM + BiGRU", [
    ["Conv1d k=3", "Conv1d k=5", ("state enc.", "feat")], ["FiLM γ·c + β (γ, β from state)"],
    ["BiGRU ×2 (h₀ from state)"], ["GRB → 128"]], kind_rows=["op", "op", "op", "op"], head=HEAD1)
card(xs[3], y1, cw, ch, "#4 transformer_moe", "Self-attention across steps + 4-expert routing", [
    ["[STATE] + 40 step tokens + positions"], ["Transformer encoder ×2 (4 heads)"],
    ["expert 1", "expert 2", "expert 3", "expert 4"], ["softmax routing + residual"]],
    kind_rows=["feat", "op", "op", "op"], head=HEAD1)
card(xs[4], y1, cw, ch, "#5 lgbm_tide_corrector", "Learns residuals on LightGBM OOF forecasts", [
    [("LightGBM q₁₉·log p", "lgb"), ("X_step", "feat")], ["TiDE (1 encoder layer) → Δ (zero init, gate 0.15)"],
    [("median + Δ₀", "q"), ("gaps × e^(0.6·tanhΔ)", "q"), ("logits + Δ", "dir")]],
    note="Starts equal to LightGBM; block OOF in training,\nfull model at test")

card(xs[0], y2, cw, ch, "#6 kan_spline_net", "Kolmogorov-Arnold RBF splines", [
    ["[raw_in ‖ 40-step context mean / std]"], ["RBF spline layer (10 knots)", "RBF spline layer (10 knots)"],
    ["GRB ×2 → h_med", "GRB ×2 → h_dist"]], kind_rows=["feat", "op", "op"], head=HEAD2)
card(xs[1], y2, cw, ch, "#7 patch_tst_horizon", "1-hour patches of 4 steps, 10 in all", [
    ["[STATE] + 10 patch tokens"], ["Transformer ×2 (med)", "Transformer ×2 (dist)"],
    ["unpatch → 40 steps → GRB", "unpatch → 40 steps → GRB"]], kind_rows=["feat", "op", "op"], head=HEAD2)
card(xs[2], y2, cw, ch, "#8 itransformer_variate", "Each feature's 40-step curve = one token", [
    ["Ds state tokens + Dh variable tokens"], ["cross-variable attention ×2"],
    ["project to 40 steps → GRB", "project to 40 steps → GRB"]], kind_rows=["feat", "op", "op"], head=HEAD2)
card(xs[3], y2, cw, ch, "#9 decoupled_tide_l1", "Two separate TiDE towers", [
    ["X_state, X_step"], ["TiDE tower (median)", "TiDE tower (distribution)"], ["h_med", "h_dist"]],
    kind_rows=["feat", "op", "feat"], head=HEAD2)
# GRB legend card
gx = xs[4]
parts.append(f'<rect x="{gx}" y="{y2}" width="{cw}" height="{ch}" rx="10" fill="#fcfcfd" stroke="#7950f2" stroke-width="1.6" stroke-dasharray="6 4"/>')
text(gx + 12, y2 + 22, "GatedResBlock (GRB, shared building block)", 14, weight="700", color="#5f3dc4")
box(gx + 100, y2 + 40, 158, 28, "feat", ["x"], size=12)
box(gx + 30, y2 + 82, 170, 28, "op", ["LayerNorm"], size=11.5)
box(gx + 30, y2 + 120, 170, 28, "op", ["Linear → SiLU"], size=11.5)
box(gx + 30, y2 + 158, 170, 28, "op", ["Linear → Dropout"], size=11.5)
box(gx + 220, y2 + 120, 110, 28, "op", ["skip / Linear"], size=11)
box(gx + 100, y2 + 196, 158, 24, "feat", ["⊕ output"], size=11.5)
arrow(gx + 150, y2 + 68, gx + 115, y2 + 80)
arrow(gx + 115, y2 + 110, gx + 115, y2 + 118)
arrow(gx + 115, y2 + 148, gx + 115, y2 + 156)
arrow(gx + 115, y2 + 186, gx + 150, y2 + 194)
arrow(gx + 210, y2 + 68, gx + 275, y2 + 118)
arrow(gx + 275, y2 + 148, gx + 210, y2 + 194)

# ---------------------------------------------------------------------------
# Section D: training pipeline
# ---------------------------------------------------------------------------
section(20, 1664, 1860, 150, "D · Training and inference (per fold, per architecture)")
dy = 1712
steps = [
    ("data", ["training origins before cutoff"]),
    ("op", ["Pass 1: inner training", "< cutoff − 14 days, 3 seeds"]),
    ("op", ["14-day hold-out: early stop", "best_epoch + interval width scaling"]),
    ("op", ["Pass 2: full refit", "fixed best_epoch, 3 new seeds"]),
    ("op", ["forecast 40 steps", "seed average → band interval scaling"]),
    ("op", ["spike prob. = ½(classifier + CDF)", "P(spread > 50), P(spread < −50)"]),
    ("loss", ["finalize → validate_submission", "→ score.py"]),
]
sw_, gap_ = 240, 22
for i, (k, lines) in enumerate(steps):
    sx = 40 + i * (sw_ + gap_)
    box(sx, dy, sw_, 56, k, lines, size=11.5, bold_first=True)
    if i:
        arrow(sx - gap_, dy + 28, sx - 2, dy + 28)
text(40, dy + 86, "Optimiser: AdamW (lr 1.5e-3, wd 1.5e-3) · cosine decay · gradient clip 1.0 · EMA 0.98/0.99 · batch 32/128 origins · at most 38 epochs · labels and scaler use only data published before the cutoff (no leakage)",
     12.5, color="#495057")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="{FONT}">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#495057"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(parts) + "</svg>")
with open(OUT, "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote", OUT, len(svg))
