"""Second jeu de figures: dispositif, interaction, intervalles, classes, tailles d'effet.

Complete figures.py. Les quatre premieres figures couvraient les courbes, les donnees et les
resultats bruts; celles-ci portent le raisonnement statistique -- ce qui manquait le plus a
la soutenance.
"""

import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ENCRE = "#1A233A"
TEAL = "#0F766E"
BRIQUE = "#B43D2E"
GRIS = "#8A93A3"
CLAIR = "#C9D3DC"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRIS, "axes.labelcolor": ENCRE,
    "xtick.color": ENCRE, "ytick.color": ENCRE, "text.color": ENCRE,
    "grid.color": "#E4E8EE", "grid.linewidth": 0.8,
})

RACINE = pathlib.Path("results/kaggle")
SORTIE = pathlib.Path(sys.argv[1])
SORTIE.mkdir(parents=True, exist_ok=True)

uh = json.loads((RACINE / "E4_uhura/resultats/E4_uhura.json").read_text(encoding="utf-8"))
mg = json.loads((RACINE / "E4_afrimgsm/resultats/E4_afrimgsm.json").read_text(encoding="utf-8"))
ah = json.loads((RACINE / "E4_afrihate_v2/resultats/E4_afrihate.json").read_text(encoding="utf-8"))
bs = json.loads((RACINE / "E4_afrihate_v2/resultats/E4_afrihate_bootstrap.json").read_text(encoding="utf-8"))


def mcnemar(a, b):
    """Renvoie (discordants, seulement_a, seulement_b, p)."""
    sa = sum(1 for x, y in zip(a, b) if x and not y)
    sb = sum(1 for x, y in zip(a, b) if y and not x)
    d = sa + sb
    if d == 0:
        return 0, 0, 0, 1.0
    k = min(sa, sb)
    p = min(1.0, 2 * sum(math.comb(d, i) for i in range(k + 1)) * 0.5 ** d)
    return d, sa, sb, p


def seuil_detectable(discordants, n):
    """Plus petit ecart net, en points, qu'un McNemar aurait declare significatif."""
    if discordants == 0:
        return float("nan")
    for k in range(discordants // 2, discordants + 1):
        p = 2 * sum(math.comb(discordants, i) for i in range(discordants - k + 1)) * 0.5 ** discordants
        if p < 0.05:
            return (2 * k - discordants) / n
    return float("nan")


# ═════════════════════════════════════════ 5 · dispositif experimental
fig, ax = plt.subplots(figsize=(13.5, 5.2))
ax.axis("off")

def boite(x, y, w, h, texte, sous, couleur, remplissage="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                linewidth=2, edgecolor=couleur, facecolor=remplissage))
    ax.text(x + w / 2, y + h * .62, texte, ha="center", fontsize=14,
            fontweight="bold", color=ENCRE)
    ax.text(x + w / 2, y + h * .24, sous, ha="center", fontsize=10.5, color=GRIS)

def fleche(x1, y, x2, etiquette, detail):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=18,
                                 linewidth=1.8, color=GRIS))
    ax.text((x1 + x2) / 2, y + .17, etiquette, ha="center", fontsize=12,
            fontweight="bold", color=ENCRE)
    ax.text((x1 + x2) / 2, y - .30, detail, ha="center", fontsize=9.5, color=GRIS)

for y, (b0, b1, b2), couleur in [
    (2.55, ("A0", "A2s", "A2d"), BRIQUE),
    (0.65, ("A1", "A3s", "A3d"), TEAL),
]:
    nom = "Qwen3.5-4B-Base" if couleur == BRIQUE else "AfriqueQwen-50Langs"
    boite(0.2, y, 3.0, 1.0, b0, nom, couleur)
    boite(5.0, y, 2.6, 1.0, b1, "+ SFT", couleur)
    boite(9.6, y, 2.6, 1.0, b2, "+ SFT + DPO", couleur, remplissage="#F4F8F8")
    fleche(3.35, y + .5, 4.9, "SFT", "2 810 Aya examples")
    fleche(7.75, y + .5, 9.5, "DPO", "709 preference pairs")

# Les fleches de comparaison tiennent dans l'espace ENTRE les deux rangees (1,65 a 2,55).
# Les faire partir du centre des boites les ferait passer par-dessus leur texte.
for x, etiquette in [(1.7, "starting gap"), (6.3, "after SFT only"), (10.9, "after SFT + DPO")]:
    ax.annotate("", xy=(x, 1.72), xytext=(x, 2.48),
                arrowprops=dict(arrowstyle="<->", color=ENCRE, lw=1.6, ls=(0, (4, 3))))
    ax.text(x, 2.10, etiquette, fontsize=11, color=ENCRE, ha="center", va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none"))

ax.text(0.2, -0.35, "Only the backbone differs. Same data, same seed, same recipe, "
        "same evaluation split (verified by MD5), same tokenizer (87 694 tokens, identical).",
        fontsize=11.5, color=ENCRE)
ax.set_xlim(-0.1, 12.6); ax.set_ylim(-0.7, 4.05)
ax.set_title("Experimental design — six states, one variable, three comparisons",
             fontsize=17, fontweight="bold", loc="left", pad=14)
fig.tight_layout()
fig.savefig(SORTIE / "5_design.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═════════════════════════════════════════ 6 · l'interaction
r = ah["resultats"]
gain_a2 = bs["gain de A2 par l'alignement  A2d - A0"]
gain_a3 = bs["gain de A3 par l'alignement  A3d - A1"]
inter = bs["INTERACTION  (A3d-A1) - (A2d-A0)"]

fig, (g, d) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                           gridspec_kw={"width_ratios": [1.25, 1]})

x = [0, 1]
a2 = [r["A0_base"]["macro_f1"], r["A2d_sft_dpo"]["macro_f1"]]
a3 = [r["A1_base"]["macro_f1"], r["A3d_sft_dpo"]["macro_f1"]]
g.plot(x, a2, color=BRIQUE, lw=3, marker="o", ms=11, label="A2 · Qwen-Base")
g.plot(x, a3, color=TEAL, lw=3, marker="o", ms=11, label="A3 · AfriqueQwen")
for xi, v in zip(x, a2):
    g.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, -22),
               ha="center", fontsize=12, color=BRIQUE, fontweight="bold")
for xi, v in zip(x, a3):
    g.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 14),
               ha="center", fontsize=12, color=TEAL, fontweight="bold")
g.axhline(ah["plancher_macro_f1"], color=GRIS, ls="--", lw=1.3)
g.text(1.04, ah["plancher_macro_f1"], "majority-class floor", fontsize=9.5,
       color=GRIS, va="center")
g.set_xticks(x); g.set_xticklabels(["raw backbone", "after SFT + DPO"], fontsize=12)
g.set_xlim(-0.28, 1.42); g.set_ylabel("macro F1 · AfriHate")
g.set_title("Alignment moves the two arms in opposite directions", pad=12)
g.grid(axis="y", alpha=.5); g.legend(frameon=False, fontsize=11.5, loc="lower left")

quantites = [("A2\nQwen-Base", gain_a2, BRIQUE), ("A3\nAfriqueQwen", gain_a3, TEAL),
             ("interaction\nA3 − A2", inter, ENCRE)]
for i, (etiquette, q, couleur) in enumerate(quantites):
    d.errorbar(i, q["observe"], yerr=[[q["observe"] - q["ic_bas"]], [q["ic_haut"] - q["observe"]]],
               fmt="o", ms=13, color=couleur, capsize=9, capthick=2.4, lw=2.6)
    d.text(i + .22, q["observe"], f"{q['observe']:+.4f}", fontsize=12.5,
           fontweight="bold", color=couleur, va="center")
d.axhline(0, color=GRIS, lw=1.6)
d.set_xticks(range(3)); d.set_xticklabels([e for e, _, _ in quantites], fontsize=11.5)
d.set_xlim(-0.5, 2.75); d.set_ylabel("change in macro F1")
d.set_title("Paired bootstrap · 95 % intervals", pad=12)
d.grid(axis="y", alpha=.5)

fig.suptitle("The interaction is real — alignment helps the CPT backbone and hurts the base one",
             fontsize=17, fontweight="bold", y=1.02)
fig.text(0.5, -0.055,
         f"Interaction = {inter['observe']:+.4f}, 95 % CI [{inter['ic_bas']:+.4f}, "
         f"{inter['ic_haut']:+.4f}] — excludes zero.   "
         "The two arms start indistinguishable: the raw gap's interval contains zero.",
         ha="center", fontsize=12, color=ENCRE)
fig.tight_layout()
fig.savefig(SORTIE / "6_interaction.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═════════════════════════════════════════ 7 · forest plot
fig, ax = plt.subplots(figsize=(12.5, 5.0))
etiquettes = {
    "ecart au depart              A1 - A0": "Starting gap   A1 − A0",
    "ecart apres SFT              A3s - A2s": "After SFT   A3s − A2s",
    "ecart apres SFT+DPO          A3d - A2d": "After SFT + DPO   A3d − A2d",
    "gain de A2 par l'alignement  A2d - A0": "Alignment gain, base arm   A2d − A0",
    "gain de A3 par l'alignement  A3d - A1": "Alignment gain, CPT arm   A3d − A1",
    "INTERACTION  (A3d-A1) - (A2d-A0)": "INTERACTION   (A3d−A1) − (A2d−A0)",
}
cles = list(etiquettes)
for i, cle in enumerate(cles):
    q = bs[cle]
    y = len(cles) - 1 - i
    couleur = ENCRE if "INTERACTION" in cle else (TEAL if q["exclut_zero"] else GRIS)
    ax.plot([q["ic_bas"], q["ic_haut"]], [y, y], color=couleur, lw=3.4,
            solid_capstyle="butt", alpha=.45)
    ax.plot([q["ic_bas"]] * 2, [y - .13, y + .13], color=couleur, lw=2.4)
    ax.plot([q["ic_haut"]] * 2, [y - .13, y + .13], color=couleur, lw=2.4)
    ax.plot(q["observe"], y, "o", ms=12, color=couleur)
    ax.text(0.118, y, f"{q['observe']:+.4f}  [{q['ic_bas']:+.4f}, {q['ic_haut']:+.4f}]",
            fontsize=11, color=ENCRE, va="center", family="DejaVu Sans Mono")
ax.axvline(0, color=BRIQUE, lw=1.8, ls="--")
ax.set_yticks(range(len(cles)))
ax.set_yticklabels([etiquettes[c] for c in reversed(cles)], fontsize=12)
ax.set_xlim(-0.05, 0.20); ax.set_ylim(-0.6, len(cles) - .3)
ax.set_xlabel("difference in macro F1")
ax.set_title("AfriHate · every quantity with its 95 % bootstrap interval",
             fontsize=16, fontweight="bold", loc="left", pad=14)
ax.grid(axis="x", alpha=.5)
fig.text(0.5, -0.03,
         "Teal = interval excludes zero.   Grey = contains zero.   "
         "2 000 paired resamples, the same row indices drawn for all six states.",
         ha="center", fontsize=11, color=GRIS)
fig.tight_layout()
fig.savefig(SORTIE / "7_forest.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═════════════════════════════════════════ 8 · F1 par classe
fig, ax = plt.subplots(figsize=(12.5, 4.9))
classes = ah["etiquettes"]
etats = [("A0_base", "A0 raw", BRIQUE, .45), ("A2d_sft_dpo", "A2 aligned", BRIQUE, 1.0),
         ("A1_base", "A1 raw", TEAL, .45), ("A3d_sft_dpo", "A3 aligned", TEAL, 1.0)]
largeur = 0.2
for j, (cle, nom, couleur, alpha) in enumerate(etats):
    valeurs = [r[cle]["per_label"][c]["f1"] for c in classes]
    positions = [i + (j - 1.5) * largeur for i in range(len(classes))]
    barres = ax.bar(positions, valeurs, largeur * .92, label=nom, color=couleur, alpha=alpha)
    for b, v in zip(barres, valeurs):
        ax.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.2f}",
                ha="center", fontsize=9, color=ENCRE)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels([f"{c}\nsupport {r['A0_base']['per_label'][c]['support']}" for c in classes],
                   fontsize=12)
ax.set_ylabel("F1"); ax.set_ylim(0, .72)
ax.legend(frameon=False, fontsize=11.5, ncol=4, loc="upper right")
ax.grid(axis="y", alpha=.5)
ax.set_title("Where the effect lives — per-class F1 on AfriHate",
             fontsize=16, fontweight="bold", loc="left", pad=14)
fig.text(0.5, -0.06,
         "The macro F1 effect sits in the minority classes, which is where a safety axis "
         "matters — and also where noise is largest.",
         ha="center", fontsize=11.5, color=ENCRE)
fig.tight_layout()
fig.savefig(SORTIE / "8_per_class.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═════════════════════════════════════════ 9 · tailles d'effet et detectabilite
axes_info = []
for nom, donnees, cle, n in [
    ("Honest\nUhura", uh, "accuracy", uh["n"]),
    ("Helpful\nAfriMGSM", mg, "accuracy", mg["n"]),
    ("Harmless\nAfriHate", ah, "accuracy", r["A0_base"]["n"]),
]:
    j = donnees["justesse"]
    d_, _, _, p = mcnemar(j["A2d_sft_dpo"], j["A3d_sft_dpo"])
    ecart = donnees["resultats"]["A3d_sft_dpo"][cle] - donnees["resultats"]["A2d_sft_dpo"][cle]
    axes_info.append((nom, abs(ecart), seuil_detectable(d_, n), p, d_, n))

fig, ax = plt.subplots(figsize=(12.5, 4.9))
positions = range(len(axes_info))
ax.bar([p - .17 for p in positions], [a[1] for a in axes_info], .32,
       label="observed |A3 − A2|", color=TEAL)
ax.bar([p + .17 for p in positions], [a[2] for a in axes_info], .32,
       label="smallest detectable effect", color=CLAIR)
for i, (nom, obs, seuil, p, d_, n) in enumerate(axes_info):
    ax.text(i - .17, obs + .006, f"{obs:.3f}", ha="center", fontsize=11,
            fontweight="bold", color=ENCRE)
    ax.text(i + .17, seuil + .006, f"{seuil:.3f}", ha="center", fontsize=11, color=GRIS)
    ax.text(i, -.032, f"n = {n} · {d_} disagreements · p = {p:.3g}",
            ha="center", fontsize=10, color=GRIS)
ax.set_xticks(list(positions)); ax.set_xticklabels([a[0] for a in axes_info], fontsize=12.5)
ax.set_ylabel("effect size"); ax.set_ylim(-.05, max(a[2] for a in axes_info) * 1.25)
ax.legend(frameon=False, fontsize=11.5)
ax.grid(axis="y", alpha=.5)
ax.set_title("What we could have detected — observed gap against the detection floor",
             fontsize=16, fontweight="bold", loc="left", pad=14)
fig.text(0.5, -0.075,
         "A non-significant result is only informative when the measurement could have seen "
         "an effect. On Uhura it could not see less than 7 points — that is the honest limit.",
         ha="center", fontsize=11.5, color=BRIQUE)
fig.tight_layout()
fig.savefig(SORTIE / "9_effect_sizes.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ═════════════════════════════════════════ 10 · inventaire des jeux
fig, ax = plt.subplots(figsize=(13.5, 5.6))
ax.axis("off")

jeux = [
    ("Aya — Hausa", "SFT", 3512, "Apache-2.0", "natively written by speakers", TEAL),
    ("Uhura generation", "DPO", 791, "MIT", "professional human translation", TEAL),
    ("UbuntuGuard — Hausa", "DPO", 95, "CC BY 4.0 *", "African-anchored", GRIS),
    ("Uhura multiple-choice", "EVAL", 183, "MIT", "77 % discarded — contamination", BRIQUE),
    ("AfriMGSM hau", "EVAL", 250, "Apache-2.0", "never trained on", TEAL),
    ("AfriHate hau", "EVAL", 1049, "CC-BY-4.0 (mirror)", "never trained on", TEAL),
]
MAX = 3512
for i, (nom, phase, n, licence, note, couleur) in enumerate(jeux):
    y = 4.55 - i * .78
    ax.text(0, y, phase, fontsize=11, fontweight="bold", color=GRIS)
    ax.text(0.85, y, nom, fontsize=13, color=ENCRE)
    largeur = 4.4 * (n / MAX) ** 0.42
    ax.add_patch(FancyBboxPatch((5.4, y - .16), largeur, .38,
                                boxstyle="round,pad=0.01,rounding_size=0.06",
                                linewidth=0, facecolor=couleur))
    ax.text(5.4 + largeur + .16, y, f"{n:,}".replace(",", " "), fontsize=12,
            fontweight="bold", color=ENCRE, va="center")
    ax.text(10.6, y, licence, fontsize=11, color=ENCRE, va="center")
    ax.text(10.6, y - .27, note, fontsize=9.5, color=GRIS, va="center", style="italic")

ax.text(0, 0.05, "* announced by the paper, not yet confirmed by the author — "
        "the licence request was sent on 3 September and is still unanswered.",
        fontsize=10.5, color=GRIS, style="italic")
ax.text(5.4, 5.15, "volume (compressed scale)", fontsize=10, color=GRIS)
ax.text(10.6, 5.15, "licence and provenance", fontsize=10, color=GRIS)
ax.set_xlim(-0.2, 14.2); ax.set_ylim(-0.25, 5.5)
ax.set_title("Every dataset used — role, volume, licence, provenance",
             fontsize=17, fontweight="bold", loc="left", pad=14)
fig.tight_layout()
fig.savefig(SORTIE / "10_datasets.png", dpi=200, bbox_inches="tight")
plt.close(fig)

for f in sorted(SORTIE.glob("*.png")):
    print(f"  {f.name:<22} {f.stat().st_size // 1024:>4} Ko")
