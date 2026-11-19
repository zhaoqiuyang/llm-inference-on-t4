"""Generate README figures from results/raw/*.csv into results/figures/.

Usage:  python scripts/plot.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parent.parent
RAW, FIG = ROOT / "results" / "raw", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- palette (validated categorical order) & ink ---------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # blue, orange, aqua
REGIME = ["#eef4fc", "#f7f6f3", "#fdf0ea"]   # memory / transition / compute bands

# ---- model constants (see README §3.4 / §4) --------------------------------
FLOP_PER_TOKEN_7B = 2 * 7.07e9      # matmul params incl. lm_head, excl. embedding
BYTES_PER_STEP_7B = 4.5e9           # int4 body + fp16 lm_head streamed per step
PEAK_TFLOPS, PEAK_BW = 65.0, 320.0  # T4 datasheet (fp16 tensor, GB/s)
HW_TFLOPS, HW_BW = 18.8, 243.0      # measured microbenchmarks
EFF_TFLOPS, EFF_BW = 15.7, 169.0    # derived from vLLM step times

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 10.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.titlesize": 12.5, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def title(fig, main, sub):
    fig.text(0.012, 0.975, main, ha="left", va="top", fontsize=14,
             fontweight="semibold", color=INK)
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=10.5, color=INK2)


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", FIG / name)


def batch_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))


# ---- 1. 0.5B throughput vs batch -------------------------------------------
def fig_throughput_05b():
    d = pd.read_csv(RAW / "sweep_0.5b.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(top=0.80)
    title(fig, "Qwen2.5-0.5B (fp16): decode throughput vs batch size",
          "Single T4, vLLM, ~9-token prompt, up to 128 output tokens")
    ax.plot(d.batch, d.throughput_tok_s, color=S1, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    batch_axis(ax, d.batch.tolist())
    ax.set_ylim(0, 11000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Batch size (concurrent requests, log scale)")
    ax.set_ylabel("Throughput (tokens/s)")
    for b, t, txt, dy in [(1, 164, "164", 12), (256, 9832, "peak 9,832", 12),
                          (512, 9649, "9,649\nqueue-capped\n(max_num_seqs=256)", -58)]:
        ax.annotate(txt, (b, t), xytext=(8 if b == 512 else 0, dy), textcoords="offset points",
                    ha="center" if b != 512 else "right", color=INK, fontsize=9.5)
    save(fig, "01_throughput_0.5b.png")


# ---- 2. 7B step time vs batch ----------------------------------------------
def fig_step_time_7b():
    d = pd.read_csv(RAW / "sweep_7b_awq.csv")
    b = np.linspace(1, 260, 400)
    model = np.maximum(26, 22 + 0.9 * b)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.22})
    fig.subplots_adjust(top=0.80)
    title(fig, "Qwen2.5-7B AWQ int4: decode step time vs batch size",
          "Flat while memory-bound, then ~0.9 ms per extra request once compute-bound. "
          "Fixed 128 output tokens, prefix caching off.")

    for ax, (lo, hi), ylim, name in [(axes[0], (0, 66), (0, 85), "B ≤ 64 (zoom)"),
                                     (axes[1], (0, 262), (0, 270), "Full range")]:
        for (x0, x1), c in zip([(lo, 10), (10, 64), (64, hi)], REGIME):
            ax.axvspan(max(x0, lo), min(x1, hi), color=c, zorder=0, lw=0)
        m = (b >= lo) & (b <= hi)
        ax.plot(b[m], model[m], color=S2, lw=2, ls=(0, (4, 3)), zorder=2,
                label="Model: max(26, 22 + 0.9·B) ms")
        dd = d[(d.batch >= lo) & (d.batch <= hi)]
        ax.plot(dd.batch, dd.step_ms, ls="none", marker="o", markersize=7,
                color=S1, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label="Measured")
        ax.set_xlim(lo, hi); ax.set_ylim(*ylim)
        ax.set_xlabel("Batch size B")
        ax.set_title(name, fontsize=11, color=INK2, fontweight="normal")
    axes[0].set_ylabel("Step time (ms)")

    ax = axes[0]
    for x, txt in [(5.6, "memory-\nbound"), (37, "transition"), (65, "")]:
        if txt:
            ax.text(x, 80, txt, ha="center", va="top", color=INK2, fontsize=9.5)
    ax.annotate("16→20: +9.5 ms\n(16-row tile?)", (20, 46.0), xytext=(27, 18),
                textcoords="data", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax = axes[1]
    ax.text(160, 262, "compute-bound", ha="center", va="top", color=INK2, fontsize=9.5)
    ax.annotate("64: 74 ms, 864 tok/s", (64, 74.1), xytext=(10, -14),
                textcoords="offset points", ha="left", fontsize=9.5, color=INK)
    ax.text(208, 224, "256: 249 ms, 1,029 tok/s", ha="right", va="bottom",
            fontsize=9.5, color=INK)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.78))
    save(fig, "02_step_time_7b.png")


# ---- 3. Roofline ------------------------------------------------------------
def fig_roofline():
    d = pd.read_csv(RAW / "sweep_7b_awq.csv")
    ai = d.batch * FLOP_PER_TOKEN_7B / BYTES_PER_STEP_7B            # FLOP/byte
    achieved = d.throughput_tok_s * FLOP_PER_TOKEN_7B / 1e12        # TFLOPS
    x = np.logspace(0, 3.4, 400)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.subplots_adjust(top=0.82)
    title(fig, "T4 roofline: datasheet vs measured, with 7B-int4 decode steps",
          "Each dot is one batch size; arithmetic intensity grows ~3.1 FLOP/byte per request")

    roofs = [(PEAK_TFLOPS, PEAK_BW, MUTED, (0, (2, 2)), "Datasheet: 65 TFLOPS, 320 GB/s"),
             (HW_TFLOPS, HW_BW, S1, "-", "Measured hardware: 18.8 TFLOPS, 243 GB/s"),
             (EFF_TFLOPS, EFF_BW, S2, (0, (5, 3)), "vLLM effective: ~15.7 TFLOPS, ~169 GB/s")]
    for peak, bw, c, ls, lab in roofs:
        ridge = peak * 1000 / bw
        ax.plot(x, np.minimum(peak, bw * x / 1000), color=c, ls=ls, lw=2,
                label=f"{lab}  (ridge {ridge:.0f} FLOP/B)")
        ax.plot([ridge], [peak], marker="o", markersize=5, color=c, zorder=4)

    ax.plot(ai, achieved, ls="none", marker="o", markersize=8, color=S3,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5,
            label="7B-int4 decode, achieved")
    for B in [1, 8, 64, 256]:
        i = d.index[d.batch == B][0]
        ax.annotate(f"B={B}", (ai[i], achieved[i]), xytext=(6, -14),
                    textcoords="offset points", fontsize=9.5, color=INK)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 2500); ax.set_ylim(0.2, 100)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Arithmetic intensity (FLOP per byte of weights read)")
    ax.set_ylabel("Throughput (TFLOPS, log)")
    ax.legend(loc="lower right", fontsize=9.5)
    save(fig, "03_roofline.png")


# ---- 4. GEMM clock / power / temperature ------------------------------------
def fig_power_throttle():
    d = pd.read_csv(RAW / "gemm_clock_power.csv")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True,
                             gridspec_kw={"hspace": 0.38})
    fig.subplots_adjust(top=0.83)
    title(fig, "T4 under a sustained fp16 GEMM: power-capped, clock sinking",
          "8192² GEMM loop, nvidia-smi sampled every ~1 s; throttle reason 0x4 (SW Power Cap) "
          "from the 2nd sample on")

    panels = [("sm_clock_mhz", "SM clock (MHz)", (400, 640),
               [(585, "base clock 585 MHz  (boost: 1,590 MHz, off-chart)")]),
              ("power_w", "Board power (W)", (25, 75), [(70, "power limit 70 W")]),
              ("temp_c", "GPU temp (°C)", (60, 85), [])]
    for ax, (col, lab, ylim, refs) in zip(axes, panels):
        ax.plot(d.t_s, d[col], color=S1, lw=2)
        for y, t in refs:
            ax.axhline(y, color=MUTED, lw=1, ls=(0, (3, 3)))
            ax.text(d.t_s.max(), y, t, ha="right", va="bottom", fontsize=9, color=INK2)
        ax.set_ylim(*ylim)
        ax.set_title(lab, fontsize=10.5, color=INK2, fontweight="normal", pad=4)
    a0 = axes[0]
    a0.annotate(f"{d.sm_clock_mhz.iloc[1]} → {d.sm_clock_mhz.iloc[-1]} MHz",
                (d.t_s.iloc[-1], d.sm_clock_mhz.iloc[-1]), xytext=(-6, -16),
                textcoords="offset points", ha="right", fontsize=9.5, color=INK)
    axes[2].text(d.t_s.max(), 61.5, "64 → 81 °C: leakage grows, so less of the\n"
                 "fixed 70 W budget is left for clock", ha="right", va="bottom",
                 fontsize=9.5, color=INK)
    axes[-1].set_xlabel("Time (s)")
    save(fig, "04_power_throttle.png")


if __name__ == "__main__":
    fig_throughput_05b()
    fig_step_time_7b()
    fig_roofline()
    fig_power_throttle()
