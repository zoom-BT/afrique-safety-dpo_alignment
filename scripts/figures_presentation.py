"""Produit le jeu de figures pour la presentation: courbes, flux de donnees, resultats.

Trois manques identifies apres la soutenance:
  1. rien ne prouvait que les resultats non significatifs ne sont pas de la memorisation
  2. aucune courbe d'entrainement
  3. aucun chiffre sur la repartition des donnees par phase

Chaque figure repond a l'un d'eux. Legendes en anglais, comme les slides.
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ENCRE = "#1A233A"
TEAL = "#0F766E"
BRIQUE = "#B43D2E"
GRIS = "#8A93A3"
FOND = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": FOND, "axes.facecolor": FOND,
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRIS, "axes.labelcolor": ENCRE,
    "xtick.color": ENCRE, "ytick.color": ENCRE, "text.color": ENCRE,
    "grid.color": "#E4E8EE", "grid.linewidth": 0.8,
})

RACINE = pathlib.Path("results/kaggle")
SORTIE = pathlib.Path(sys.argv[1])
SORTIE.mkdir(parents=True, exist_ok=True)


def lire(chemin):
    return json.loads((RACINE / chemin).read_text(encoding="utf-8"))


def points(historique, cle):
    return [(e["step"], e[cle]) for e in historique if cle in e]


# ═══════════════════════════════════════════════ 1 · courbes de SFT
sft_a3 = lire("R1a/results/sft/training_log_history.json")
sft_a2 = lire("R2a/results/sft/training_log_history.json")
pa3, pa2 = points(sft_a3, "loss"), points(sft_a2, "loss")

fig, (g, d) = plt.subplots(1, 2, figsize=(13, 4.6))

g.plot(*zip(*pa2), color=BRIQUE, lw=2.2, label="A2 · Qwen3.5-4B-Base")
g.plot(*zip(*pa3), color=TEAL, lw=2.2, label="A3 · AfriqueQwen-50Langs")
g.fill_between([s for s, _ in pa3], [l for _, l in pa3], [l for _, l in pa2],
               color=GRIS, alpha=0.13)
g.annotate("constant offset ≈ 0.9 nats", xy=(180, 1.78), fontsize=11.5,
           color=GRIS, ha="center")
g.set_xlabel("training step"); g.set_ylabel("SFT loss")
g.set_title("Same data, same recipe — only the backbone differs", pad=12)
g.grid(alpha=.5); g.legend(frameon=False, fontsize=11.5)

descentes = {"A2\nQwen-Base": pa2[0][1] - pa2[-1][1], "A3\nAfriqueQwen": pa3[0][1] - pa3[-1][1]}
barres = d.bar(list(descentes), list(descentes.values()),
               color=[BRIQUE, TEAL], width=.55)
for b, v in zip(barres, descentes.values()):
    d.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.3f}",
           ha="center", fontsize=14, fontweight="bold", color=ENCRE)
d.set_ylim(0, max(descentes.values()) * 1.32)
d.set_ylabel("loss reduction over training")
d.set_title("Both arms learn the same amount", pad=12)
d.grid(axis="y", alpha=.5)

fig.suptitle("SFT — the CPT advantage is an offset, not an amplifier",
             fontsize=17, fontweight="bold", y=1.005)
fig.tight_layout()
fig.savefig(SORTIE / "1_sft_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════ 2 · courbes de DPO
dpo_a3 = lire("R1b/results/dpo/training_log_history.json")
dpo_a2 = lire("R2b/results/dpo/training_log_history.json")

fig, (g, d) = plt.subplots(1, 2, figsize=(13, 4.6))

g.plot(*zip(*points(dpo_a2, "loss")), color=BRIQUE, lw=2.2, label="A2 · Qwen-Base")
g.plot(*zip(*points(dpo_a3, "loss")), color=TEAL, lw=2.2, label="A3 · AfriqueQwen")
g.set_xlabel("training step"); g.set_ylabel("DPO loss")
g.set_title("DPO loss", pad=12); g.grid(alpha=.5)
g.legend(frameon=False, fontsize=11.5)

ma2 = points(dpo_a2, "rewards/margins")
ma3 = points(dpo_a3, "rewards/margins")
d.plot(*zip(*ma2), color=BRIQUE, lw=2.2, marker="o", ms=4)
d.plot(*zip(*ma3), color=TEAL, lw=2.2, marker="o", ms=4)
d.axhline(0, color=GRIS, lw=1)
d.annotate(f"×{ma3[-1][1] / ma3[0][1]:.0f}", xy=(ma3[-1][0], ma3[-1][1]),
           xytext=(-58, -6), textcoords="offset points",
           fontsize=17, fontweight="bold", color=TEAL)
d.set_xlabel("training step"); d.set_ylabel("reward margin")
d.set_title("Reward margins — the DPO did learn", pad=12); d.grid(alpha=.5)

fig.suptitle("DPO — 886 preference pairs carried enough signal",
             fontsize=17, fontweight="bold", y=1.005)
fig.text(0.5, -0.055,
         "Margins are measured against each arm's own frozen reference — "
         "they are NOT comparable across arms.",
         ha="center", fontsize=11.5, color=BRIQUE, style="italic")
fig.tight_layout()
fig.savefig(SORTIE / "2_dpo_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════ 3 · flux de donnees
fig, ax = plt.subplots(figsize=(13.5, 5.4))
ax.axis("off")

# Barres NORMALISEES: chaque phase occupe la meme largeur, les segments montrent la
# proportion. Une echelle absolue ecraserait les phases DPO et EVAL sous le SFT.
etapes = [
    ("SFT", "Aya — Hausa slice", "natively written by speakers · Apache-2.0",
     [("train  2 810", 2810, TEAL), ("held out  702", 702, "#C9D3DC")], 3512),
    ("DPO", "Uhura generation + UbuntuGuard", "human translation · MIT",
     [("train  709", 709, TEAL), ("held out  177", 177, "#C9D3DC")], 886),
    ("EVAL", "Uhura multiple-choice", "same TruthfulQA the DPO trained on",
     [("seen in DPO training  625", 625, BRIQUE), ("usable  183", 183, TEAL)], 808),
]

X0, LARGEUR = 4.9, 8.2
for i, (phase, source, note, segments, total) in enumerate(etapes):
    y = 3.55 - i * 1.28
    ax.text(0, y + .40, phase, fontsize=16, fontweight="bold", color=ENCRE)
    ax.text(0, y + .06, f"{total:,}".replace(",", " ") + " items", fontsize=12, color=GRIS)
    ax.text(1.35, y + .42, source, fontsize=13, color=ENCRE)
    ax.text(1.35, y + .10, note, fontsize=10.5, color=GRIS, style="italic")
    x = X0
    for etiquette, valeur, couleur in segments:
        w = LARGEUR * valeur / total
        ax.add_patch(Rectangle((x, y), w, .50, color=couleur))
        ax.text(x + w / 2, y + .25, etiquette, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white" if couleur != "#C9D3DC" else ENCRE)
        ax.text(x + w / 2, y - .20, f"{100*valeur/total:.0f} %", ha="center",
                fontsize=10, color=GRIS)
        x += w

ax.text(0, 0.30, "Why 625 questions had to be discarded", fontsize=13,
        fontweight="bold", color=BRIQUE)
ax.text(0, 0.00,
        "Uhura publishes its generation and multiple-choice configs from the same TruthfulQA, "
        "and the DPO trains on the first.",
        fontsize=11.5, color=ENCRE)
ax.text(0, -0.28,
        "77 % of the evaluation set had therefore already been seen. "
        "Every split is held out at question level, verified 0 overlap.",
        fontsize=11.5, color=ENCRE)
ax.set_xlim(-0.2, 13.4); ax.set_ylim(-0.55, 4.35)
ax.set_title("Data at each stage — volumes, provenance, and what had to be discarded",
             fontsize=17, fontweight="bold", loc="left", pad=14)
fig.tight_layout()
fig.savefig(SORTIE / "3_data_flow.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════ 4 · resultats E4
uh = lire("E4_uhura/resultats/E4_uhura.json")
mg = lire("E4_afrimgsm/resultats/E4_afrimgsm.json")
ah = lire("E4_afrihate/resultats/E4_afrihate.json")

ORDRE = ["A0_base", "A1_base", "A2s_sft", "A3s_sft", "A2d_sft_dpo", "A3d_sft_dpo"]
COURT = ["A0\nraw", "A1\nraw", "A2\n+SFT", "A3\n+SFT", "A2\n+DPO", "A3\n+DPO"]
COULEURS = [BRIQUE, TEAL] * 3

panneaux = [
    ("Honest · Uhura MCQ", uh, "accuracy", uh["resultats"]["A0_base"]["random_baseline"],
     "chance", f"n = {uh['n']} uncontaminated"),
    ("Helpful · AfriMGSM", mg, "accuracy", None, None, f"n = {mg['n']}"),
    ("Harmless · AfriHate", ah, "macro_f1", ah["plancher_macro_f1"],
     "majority class", f"n = {ah['resultats']['A0_base']['n']}"),
]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
for ax, (titre, donnees, cle, plancher, nom_plancher, sous) in zip(axes, panneaux):
    valeurs = [donnees["resultats"][e][cle] for e in ORDRE]
    barres = ax.bar(range(6), valeurs, color=COULEURS, width=.68)
    for b, v in zip(barres, valeurs):
        ax.text(b.get_x() + b.get_width() / 2, v + max(valeurs) * .022, f"{v:.3f}",
                ha="center", fontsize=10, color=ENCRE)
    if plancher:
        ax.axhline(plancher, color=GRIS, ls="--", lw=1.4)
        ax.text(5.45, plancher + max(valeurs) * .015, nom_plancher, ha="right",
                fontsize=9.5, color=GRIS)
    ax.set_xticks(range(6)); ax.set_xticklabels(COURT, fontsize=9.5)
    ax.set_ylim(0, max(valeurs) * 1.22)
    ax.set_title(titre, pad=10)
    ax.text(0.5, -0.235, sous, transform=ax.transAxes, ha="center",
            fontsize=10, color=GRIS)
    ax.grid(axis="y", alpha=.5)
axes[0].set_ylabel("score")

fig.suptitle("E4 — six model states, three axes, one variable",
             fontsize=17, fontweight="bold", y=1.02)
fig.text(0.5, -0.085,
         "Red = Qwen-Base arm   ·   Teal = AfriqueQwen arm.   "
         "Every A3 − A2 gap is statistically indistinguishable from zero, "
         "or already present before any training.",
         ha="center", fontsize=11.5, color=ENCRE)
fig.tight_layout()
fig.savefig(SORTIE / "4_e4_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)

for f in sorted(SORTIE.glob("*.png")):
    print(f"  {f.name:<22} {f.stat().st_size // 1024:>4} Ko")
