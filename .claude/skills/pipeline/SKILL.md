---
name: pipeline
description: Comment reproduire seul la chaine complete du projet, du telechargement des donnees jusqu aux figures. A lire avant de relancer un run, de changer une graine, ou de reprendre le projet apres une pause.
---

# Reproduire la chaine, seul

Ce fichier decrit la chaine reelle, celle qui a produit les resultats. Il ne decrit pas une
chaine ideale. Chaque regle enoncee ici vient d une panne constatee.

## Le principe qui commande tout le reste

Le depot est la source. Colab et Kaggle sont des executants.

Une session Colab a ete reinitialisee un jour ou les cellules E2 et E3 avaient ete ecrites
directement dans le navigateur. Tout a disparu. Depuis, aucun code ne vit ailleurs que dans
`notebooks/` ou `src/`, et les notebooks sont generes par un script plutot que tapes a la
main, pour qu ils restent reproductibles.

## Ce que fait chaque fichier

### Configuration et chemins

`config.yaml` porte tous les reglages numeriques. Rien n est code en dur ailleurs.

`src/paths.py` trouve le code et les donnees ou qu ils soient montes. Kaggle ne monte pas
tous les datasets a la meme profondeur: un kernel a montre `/kaggle/input/<slug>/src/`, un
autre `/kaggle/input/datasets/<user>/<slug>/src/`. La recherche est donc recursive, jamais un
chemin fixe.

### Construction des donnees

`src/data.py` fabrique les exemples a partir des sources brutes.

* `build_aya_sft_examples` produit les demonstrations du SFT.
* `build_uhura_pairs` et `build_preference_pairs` produisent les paires du DPO.
* `split_by_base_stem` decoupe sans contamination, en regroupant par question de base.

### Entrainement

`src/train.py` contient `run_sft` et `run_dpo`. Les deux recoivent leurs exemples en
argument au lieu de les charger eux memes: l appelant a deja fait le decoupage, et le refaire
avec une autre graine reintroduirait la contamination.

### Evaluation

`src/eval_mcq.py` score un questionnaire a choix multiples par log vraisemblance des options.
`src/eval_tasks.py` fait de meme pour une reponse numerique et pour une classification.
Aucun modele n evalue jamais le texte d un autre modele.

### Soumission

`scripts/kaggle_run.py` pousse un notebook, suit son statut, rapatrie ses sorties, et publie
les datasets.

## L ordre d execution

```
1.  SFT bras A3        notebooks/05_R1a_sft.ipynb          environ 5 h
2.  SFT bras A2        notebooks/07_R2a_sft_controle.ipynb environ 5 h
3.  publier les adaptateurs SFT
4.  DPO bras A3        notebooks/08_R1b_dpo.ipynb          environ 3,5 h
5.  DPO bras A2        notebooks/09_R2b_dpo.ipynb          environ 3,8 h
6.  publier les adaptateurs DPO
7.  E4 sur trois axes  notebooks/10, 11, 12                environ 9 h
8.  figures            scripts/figures_presentation*.py    sans GPU
```

Les etapes 1 et 2 tournent en parallele, de meme que 4 et 5. Kaggle autorise exactement deux
sessions GPU simultanees; une troisieme est refusee.

## Les commandes

```bash
# publier le code avant tout run qui en depend
python scripts/kaggle_run.py dataset code -m "message"

# pousser un notebook
python scripts/kaggle_run.py push notebooks/05_R1a_sft.ipynb --accelerator t4 \
    --dataset afrique-safety-dpo-code --dataset afrique-safety-dpo-data

# suivre
python scripts/kaggle_run.py status 05-r1a-sft

# rapatrier
python scripts/kaggle_run.py fetch 05-r1a-sft
```

## Les pieges, et pourquoi ils comptent

Tous produisent un resultat faux sans lever d exception. C est ce qui les rend dangereux.

**Publier le dataset de code avant de pousser un notebook.** Sinon le run part sur du code
perime et personne ne le signale.

**Un adaptateur pose sur le mauvais backbone ne leve rien.** Il produit du bruit. Chaque
`adapter_config.json` declare sa base, et les notebooks comparent cette declaration au modele
qu ils chargent.

**Kaggle supprime le dossier de tete du zip quand il est seul a la racine.** On envoie
`adapters/A3_s42_sft/`, on recoit `A3_s42_sft/`. Chercher par nom de dossier, jamais par
prefixe.

**Les listes de l API Kaggle sont paginees a vingt entrees.** Suivre `next_page_token`, sinon
un fichier bien present parait absent.

**Python garde en cache un module deja importe.** Un `git pull` ne le remplace pas. Recharger
explicitement avec `importlib.reload`, sinon la cellule tourne sur la version d avant.

**La graine d entrainement et la graine de partition sont deux choses distinctes.**
`GRAINE` fait varier l initialisation et l ordre des exemples. `GRAINE_SPLIT` decide qui est
en evaluation, et reste fixe a 42. Les faire varier ensemble melangerait deux sources de
variance.

**La recette est figee pour toutes les graines d une meme etude.** Optimiser entre deux
graines rendrait leur ecart ininterpretable.

**Conserver la justesse question par question.** Sans elle, aucun test apparie n est possible.
Cette information a ete jetee une premiere fois comme trop volumineuse, ce qui interdisait le
seul test capable de trancher.

## Verifications a refaire apres chaque run

```bash
# la partition d evaluation est elle identique entre les bras
md5sum results/kaggle/R1a/results/sft_eval_stems_split42.json \
       results/kaggle/R2a/results/sft_eval_stems_split42.json

# la reference du DPO est elle bien l adaptateur SFT
python -c "
from safetensors.torch import load_file
a = load_file('adapters/A3_s42_sft/adapter_model.safetensors')
b = load_file('results/kaggle/R1b/results/dpo_checkpoints/final/ref/adapter_model.safetensors')
print(max((a[k].float() - b[k].float()).abs().max().item() for k in a))
"
```

La premiere doit donner deux empreintes identiques. La seconde doit donner zero.

## Les tests

`python -m pytest tests/ -q` doit passer avant tout run. Un test en particulier,
`test_every_sft_argument_exists_in_the_installed_trl`, verifie que chaque argument passe au
`SFTConfig` existe dans le TRL installe. Il a deja intercepte un argument disparu entre deux
versions, avant qu un run ne parte.
