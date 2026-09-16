"""Schemas isometriques pour le chapitre de generalites.

Projection isometrique classique: un point (x, y, z) devient (u, v) avec
    u = (x - y) * cos(30 deg)
    v = (x + y) * sin(30 deg) + z
Chaque bloc a trois faces visibles, ombrees differemment pour donner le relief: dessus
clair, droite moyenne, gauche foncee.

Une pile entiere se decale en coordonnees d ECRAN, jamais en x de scene: en isometrie,
avancer en x fait aussi monter, et deux piles decalees en x ne seraient pas de niveau.

Reserve aux schemas de concepts. Les figures de donnees restent plates.
"""

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))
ENCRE, GRIS = "#1A233A", "#8A93A3"


def iso(x, y, z):
    return (x - y) * C30, (x + y) * S30 + z


def teinte(hexa, facteur):
    """Eclaircit (facteur > 1) ou assombrit (< 1) une couleur hexadecimale."""
    r, g, b = (int(hexa[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (min(255, int(c * facteur)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def pile(ax, du, couleur, nom_base, couches):
    """Une pile de blocs, decalee de `du` en coordonnees d ecran pour rester de niveau."""
    W, D = 2.3, 1.7
    z = 0.0
    sommets = []
    for i, (nom, h) in enumerate(couches):
        col = couleur if nom == "base" else teinte(couleur, 1.0 + 0.24 * i)
        bord = teinte(couleur, .55)
        faces = {
            "gauche": [iso(0, 0, z), iso(W, 0, z), iso(W, 0, z + h), iso(0, 0, z + h)],
            "droite": [iso(W, 0, z), iso(W, D, z), iso(W, D, z + h), iso(W, 0, z + h)],
            "dessus": [iso(0, 0, z + h), iso(W, 0, z + h), iso(W, D, z + h), iso(0, D, z + h)],
        }
        for face, f in (("gauche", .70), ("droite", .85), ("dessus", 1.12)):
            pts = [(u + du, v) for u, v in faces[face]]
            ax.add_patch(Polygon(pts, closed=True, facecolor=teinte(col, f),
                                 edgecolor=bord, linewidth=1.1, zorder=10 + i))
        # Le nom du backbone ne tient sur aucune face: la face du dessus est couverte
        # par le bloc suivant, et la face avant est un parallelogramme trop etroit. Il
        # est ecrit sous la pile, avec le nom du bras.
        if nom != "base":
            cx, cy = iso(W / 2, D / 2, z + h)
            ax.text(cx + du, cy, f"+ {nom}", ha="center", va="center", fontsize=13.5,
                    color="white", fontweight="bold", zorder=60)
        sommets.append((z, h))
        z += h + 0.10
    return sommets, W, D


S = pathlib.Path(sys.argv[1]); S.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_aspect("equal"); ax.axis("off")

couches = [("base", 1.05), ("SFT", 0.75), ("DPO", 0.75)]
ROUGE, VERT = "#B43D2E", "#0F766E"
DECALAGE = 8.2

sommets, W, D = pile(ax, 0.0, ROUGE, "Qwen3.5-4B-Base", couches)
pile(ax, DECALAGE, VERT, "AfriqueQwen-50Langs", couches)

SAUT = chr(10)
for du, bras, backbone, mention in ((0.0, "bras A2", "Qwen3.5-4B-Base", "sans pre-entrainement continu"),
                                    (DECALAGE, "bras A3", "AfriqueQwen-50Langs", "avec pre-entrainement continu")):
    cx, cy = iso(W / 2, D / 2, 0)
    ax.text(cx + du, cy - 1.45, bras, ha="center", va="top", fontsize=12.5, color=ENCRE,
            fontweight="bold")
    ax.text(cx + du, cy - 1.95, backbone, ha="center", va="top", fontsize=12, color=ENCRE,
            family="DejaVu Sans Mono")
    ax.text(cx + du, cy - 2.42, mention, ha="center", va="top", fontsize=11, color=GRIS)

etats = ["A0  /  A1", "A2s  /  A3s", "A2d  /  A3d"]
for (z, h), nom in zip(sommets, etats):
    cx, cy = iso(0, 0, z + h / 2)
    ax.text(cx - 1.15, cy, nom, ha="right", va="center", fontsize=11.5,
            color=GRIS, family="DejaVu Sans Mono")

legendes = ["modele base : complete du texte,\nne suit aucune instruction",
            "SFT : apprend un format\nsur 2 810 demonstrations",
            "DPO : apprend une preference\nsur 709 paires"]
for (z, h), texte in zip(sommets, legendes):
    cx, cy = iso(W, D, z + h / 2)
    ax.text(cx + DECALAGE + 2.4, cy, texte, ha="left", va="center", fontsize=11.5, color=ENCRE)

fig.suptitle("Une couche d alignement posee sur deux backbones differents",
             fontsize=16.5, fontweight="bold", color=ENCRE, x=0.5, y=0.97)
fig.text(0.5, 0.905, "Memes donnees, meme recette, meme graine a chaque etage. "
         "Seul le bloc du dessous change.", ha="center", fontsize=12, color=GRIS)

ax.set_xlim(-5.4, 18.0); ax.set_ylim(-3.4, 5.2)
fig.savefig(S / "fig_02_chaine_alignement_iso.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("ecrit :", S / "fig_02_chaine_alignement_iso.png")
