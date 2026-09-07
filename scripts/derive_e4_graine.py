"""Derive les notebooks E4 de la graine 43.

A0 et A1 sont des modeles BRUTS: aucune graine n'intervient dans leur production, donc les
reevaluer par graine serait une depense pure. Ils ne sont conserves que sur AfriHate, ou le
calcul de l'interaction en a besoin comme point "avant".
"""
import ast
import json
import pathlib

PLANS = [
    ("notebooks/10_E4_uhura.ipynb",    "notebooks/18_E4_uhura_s43.ipynb",    False, "E4_uhura"),
    ("notebooks/11_E4_afrimgsm.ipynb", "notebooks/19_E4_afrimgsm_s43.ipynb", False, "E4_afrimgsm"),
    ("notebooks/12_E4_afrihate.ipynb", "notebooks/20_E4_afrihate_s43.ipynb", True,  "E4_afrihate"),
]

BLOC_42 = '''ETATS = [
    ("A0_base",       QWEN,    None),
    ("A1_base",       AFRIQUE, None),
    ("A2s_sft",       QWEN,    "A2_s42_sft"),
    ("A3s_sft",       AFRIQUE, "A3_s42_sft"),
    ("A2d_sft_dpo",   QWEN,    "A2_s42_dpo"),
    ("A3d_sft_dpo",   AFRIQUE, "A3_s42_dpo"),
]'''

ENTRAINES = [
    '    ("A2s_sft",       QWEN,    "A2_s43_sft"),',
    '    ("A3s_sft",       AFRIQUE, "A3_s43_sft"),',
    '    ("A2d_sft_dpo",   QWEN,    "A2_s43_dpo"),',
    '    ("A3d_sft_dpo",   AFRIQUE, "A3_s43_dpo"),',
]
BRUTS = [
    '    ("A0_base",       QWEN,    None),',
    '    ("A1_base",       AFRIQUE, None),',
]
ADAPTATEURS_42 = ("A2_s42_sft", "A3_s42_sft", "A2_s42_dpo", "A3_s42_dpo")

for source, cible, avec_bruts, prefixe in PLANS:
    nb = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
    remplace = "\n".join(["ETATS = ["] + (BRUTS if avec_bruts else []) + ENTRAINES + ["]"])

    fait = {"etats": False, "sortie": False}
    for c in nb["cells"]:
        avant = "".join(c["source"])
        apres = avant
        if BLOC_42 in apres:
            apres = apres.replace(BLOC_42, remplace)
            fait["etats"] = True
        if '"' + prefixe + '.json"' in apres:
            apres = apres.replace('"' + prefixe + '.json"', '"' + prefixe + '_s43.json"')
            fait["sortie"] = True
        if apres != avant:
            parts = apres.split("\n")
            c["source"] = [p + "\n" for p in parts[:-1]] + parts[-1:]

    manquants = [k for k, v in fait.items() if not v]
    assert not manquants, (source, manquants)

    entier = "".join("".join(c["source"]) for c in nb["cells"])
    # Seuls les noms d'adaptateurs ENTRE GUILLEMETS pilotent quelque chose; un "_s42_"
    # apparaissant dans une docstring n'est qu'un exemple illustratif.
    cites = [a for a in ADAPTATEURS_42 if '"' + a + '"' in entier]
    assert not cites, (source, "adaptateurs graine 42 encore cites", cites)
    assert entier.count("_s43_") == 4, entier.count("_s43_")

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            ast.parse("\n".join(l for l in s.split("\n") if not l.lstrip().startswith("!")))

    pathlib.Path(cible).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{cible}  {6 if avec_bruts else 4} etats  ->  {prefixe}_s43.json")
