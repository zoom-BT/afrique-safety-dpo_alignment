"""Figures de comparaison entre les graines 42 et 43.

Titres seuls, pas de legende bavarde, et aucun tiret cadratin dans le texte des figures.
"""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENCRE, TEAL, BRIQUE, GRIS = "#1A233A", "#0F766E", "#B43D2E", "#8A93A3"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 16,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRIS, "axes.labelcolor": ENCRE,
    "xtick.color": ENCRE, "ytick.color": ENCRE, "text.color": ENCRE,
    "grid.color": "#E4E8EE",
})
R = pathlib.Path("results/kaggle")
S = pathlib.Path(sys.argv[1]); S.mkdir(parents=True, exist_ok=True)
lire = lambda p: json.loads((R / p).read_text(encoding="utf-8"))
pts = lambda h, k: [(e["step"], e[k]) for e in h if k in e]

# 1. courbes de SFT, deux bras, deux graines
fig, ax = plt.subplots(figsize=(11, 5))
for chemin, coul, style, lab in (
    ("R2a/results/sft/training_log_history.json",     BRIQUE, "-",  "A2 seed 42"),
    ("R2a_s43/results/sft/training_log_history.json", BRIQUE, "--", "A2 seed 43"),
    ("R1a/results/sft/training_log_history.json",     TEAL,   "-",  "A3 seed 42"),
    ("R1a_s43/results/sft/training_log_history.json", TEAL,   "--", "A3 seed 43"),
):
    ax.plot(*zip(*pts(lire(chemin), "loss")), color=coul, ls=style, lw=2.2, label=lab)
ax.set_xlabel("step"); ax.set_ylabel("SFT loss")
ax.legend(frameon=False, ncol=2); ax.grid(alpha=.5)
ax.set_title("SFT loss reproduces across seeds", fontweight="bold", pad=14)
fig.tight_layout(); fig.savefig(S / "s43_1_sft.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 2. marges de DPO
fig, ax = plt.subplots(figsize=(11, 5))
for chemin, coul, style, lab in (
    ("R2b/results/dpo/training_log_history.json",     BRIQUE, "-",  "A2 seed 42"),
    ("R2b_s43/results/dpo/training_log_history.json", BRIQUE, "--", "A2 seed 43"),
    ("R1b/results/dpo/training_log_history.json",     TEAL,   "-",  "A3 seed 42"),
    ("R1b_s43/results/dpo/training_log_history.json", TEAL,   "--", "A3 seed 43"),
):
    ax.plot(*zip(*pts(lire(chemin), "rewards/margins")), color=coul, ls=style,
            lw=2.2, marker="o", ms=4, label=lab)
ax.set_xlabel("step"); ax.set_ylabel("reward margin")
ax.legend(frameon=False, ncol=2); ax.grid(alpha=.5)
ax.set_title("Margins track together, then diverge in the final steps",
             fontweight="bold", pad=14)
fig.tight_layout(); fig.savefig(S / "s43_2_dpo.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 3. les trois axes, ecart A3 moins A2 apres alignement
uh42, uh43 = lire("E4_uhura/resultats/E4_uhura.json"), lire("uhura_s43/resultats/E4_uhura_s43.json")
mg42, mg43 = lire("E4_afrimgsm/resultats/E4_afrimgsm.json"), lire("afrimgsm_s43/resultats/E4_afrimgsm_s43.json")
ah42, ah43 = lire("E4_afrihate_v2/resultats/E4_afrihate.json"), lire("afrihate_s43/resultats/E4_afrihate_s43.json")
ecart = lambda d, c: d["resultats"]["A3d_sft_dpo"][c] - d["resultats"]["A2d_sft_dpo"][c]
axes_n = ["Honest\nUhura", "Helpful\nAfriMGSM", "Harmless\nAfriHate"]
v42 = [ecart(uh42, "accuracy"), ecart(mg42, "accuracy"), ecart(ah42, "macro_f1")]
v43 = [ecart(uh43, "accuracy"), ecart(mg43, "accuracy"), ecart(ah43, "macro_f1")]

fig, ax = plt.subplots(figsize=(11, 5))
x = range(3)
ax.bar([i - .18 for i in x], v42, .34, color=TEAL, label="seed 42")
ax.bar([i + .18 for i in x], v43, .34, color=TEAL, alpha=.5, label="seed 43")
for i, (a, b) in enumerate(zip(v42, v43)):
    ax.text(i - .18, a + .004, f"{a:+.3f}", ha="center", fontsize=11)
    ax.text(i + .18, b + .004, f"{b:+.3f}", ha="center", fontsize=11)
ax.axhline(0, color=GRIS, lw=1.4)
ax.set_xticks(list(x)); ax.set_xticklabels(axes_n)
ax.set_ylabel("A3 minus A2, after alignment")
ax.legend(frameon=False); ax.grid(axis="y", alpha=.5)
ax.set_title("Same verdict on all three axes", fontweight="bold", pad=14)
fig.tight_layout(); fig.savefig(S / "s43_3_axes.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 4. l interaction, intervalles des deux graines
b42 = lire("E4_afrihate_v2/resultats/E4_afrihate_bootstrap.json")
b43 = lire("afrihate_s43/resultats/E4_afrihate_bootstrap.json")
cle = "INTERACTION  (A3d-A1) - (A2d-A0)"
fig, ax = plt.subplots(figsize=(9, 4.4))
for i, (q, lab, coul) in enumerate(((b42[cle], "seed 42", TEAL), (b43[cle], "seed 43", TEAL))):
    ax.errorbar(q["observe"], i, xerr=[[q["observe"] - q["ic_bas"]], [q["ic_haut"] - q["observe"]]],
                fmt="o", ms=14, color=coul, alpha=1 if i == 0 else .55,
                capsize=10, capthick=2.6, lw=2.8)
    ax.text(q["ic_haut"] + .004, i, f"{q['observe']:+.4f}", va="center",
            fontsize=13, fontweight="bold")
ax.axvline(0, color=BRIQUE, lw=1.8, ls="--")
ax.set_yticks([0, 1]); ax.set_yticklabels(["seed 42", "seed 43"])
ax.set_ylim(-.6, 1.6); ax.set_xlim(-.01, .085)
ax.set_xlabel("interaction, macro F1"); ax.grid(axis="x", alpha=.5)
ax.set_title("The interaction replicates", fontweight="bold", pad=14)
fig.tight_layout(); fig.savefig(S / "s43_4_interaction.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 5. amplitude du changement entre graines, par type de mesure
res = lambda p, b: json.loads((R / p).read_text(encoding="utf-8"))["resultats"][b]
quantites = [
    ("DPO reward margin, A3", abs(lire("R1b/results/dpo_resume_A3_s42.json")["marge_finale"]
                                  - lire("R1b_s43/results/dpo_resume_A3_s43.json")["marge_finale"]), BRIQUE),
    ("SFT final loss, A3",    abs(lire("R1a/results/sft_resume_A3_s42.json")["loss_finale"]
                                  - lire("R1a_s43/results/sft_resume_A3_s43.json")["loss_finale"]), GRIS),
    ("AfriMGSM accuracy, A3", abs(mg42["resultats"]["A3d_sft_dpo"]["accuracy"]
                                  - mg43["resultats"]["A3d_sft_dpo"]["accuracy"]), TEAL),
    ("AfriHate interaction",  abs(b42[cle]["observe"] - b43[cle]["observe"]), TEAL),
    ("Uhura accuracy, A3",    abs(uh42["resultats"]["A3d_sft_dpo"]["accuracy"]
                                  - uh43["resultats"]["A3d_sft_dpo"]["accuracy"]), TEAL),
]
quantites.sort(key=lambda t: -t[1])
fig, ax = plt.subplots(figsize=(11, 4.8))
noms = [n for n, _, _ in quantites]
vals = [v for _, v, _ in quantites]
cols = [c for _, _, c in quantites]
b = ax.barh(range(len(noms)), vals, .6, color=cols)
for i, v in enumerate(vals):
    ax.text(v + .004, i, f"{v:.4f}", va="center", fontsize=12, fontweight="bold")
ax.set_yticks(range(len(noms))); ax.set_yticklabels(noms)
ax.invert_yaxis(); ax.set_xlim(0, max(vals) * 1.25)
ax.set_xlabel("absolute change between seeds"); ax.grid(axis="x", alpha=.5)
ax.set_title("Final training metrics move, held out measures do not",
             fontweight="bold", pad=14)
fig.tight_layout(); fig.savefig(S / "s43_5_stability.png", dpi=200, bbox_inches="tight"); plt.close(fig)

for f in sorted(S.glob("s43_*.png")):
    print(f"  {f.name:<24} {f.stat().st_size // 1024:>4} Ko")
