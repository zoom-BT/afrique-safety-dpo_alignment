# Plan du rapport

Forme reprise du rapport de Tchoumi, qui est au bon niveau (4e annee, pre-ingenieur). Fond
de recherche adapte de notre travail. Les chapitres 3 a 5 de Tchoumi portaient un projet
logiciel (methodologie de conception, implementation, resultats d interface) : ils sont ici
transposes a un dispositif experimental.

**A valider avant que je redige quoi que ce soit.**

---

## Pages liminaires

| fichier | contenu | statut |
| :---- | :---- | :---- |
| `01_page_garde.md` | logos ENSPY, Mila et UdeM, titre, mention "Rapport de stage PRE-INGENIEUR", auteur, superviseur, annee | **a completer** |
| `02_dedicaces.md` | une ligne | a vous |
| `03_remerciements.md` | encadrant, enseignants, famille | a vous |
| `04_abreviations.md` | CPT, SFT, DPO, LoRA, QLoRA, HHH, QCM, ENSPY, Mila, UdeM | pret a generer |
| `05_resume.md` | francais, avec mots cles | a rediger |
| `06_abstract.md` | anglais, meme contenu | a rediger |

---

## Introduction

Le probleme en trois temps. Les modeles de langue africains issus de pre-entrainement
continu existent et sont publics. Ce sont des modeles **base** : ils completent du texte, ne
suivent pas d instruction, n ont aucune preference alignee. Personne n a encore applique sur
eux l etape d alignement que decrit InstructGPT.

La question du stage, en une phrase : **l alignement fonctionne-t-il mieux sur un backbone
pre-entraine en continu que sur sa base d origine ?**

Annonce du plan.

---

## Chapitre 1 : Contexte et structure d accueil

Chez Tchoumi, ce chapitre presentait l entreprise. Ici il presente le cadre de recherche.

- 1.1 Mila et l Universite de Montreal : l institut, son objet, sa place dans la recherche
  en apprentissage automatique
- 1.2 L encadrement : Pascal Junior Tikeng Notsawo, et le cadre du stage a distance
- 1.3 Le sujet et sa genese, y compris la redefinition de la version 1 vers la version 2
- 1.4 Les moyens : aucun GPU local, Kaggle et Colab en gratuit, et ce que cette contrainte
  impose au dispositif
- 1.5 Le calendrier des huit semaines

Les logos de la page de garde sont donc ceux de l ENSPY, de Mila et de l Universite de
Montreal.

---

## Chapitre 2 : Generalites et concepts fondamentaux

Ce chapitre donne au lecteur les notions necessaires pour suivre la suite, dans l ordre ou
il en aura besoin. Il ne fait pas de revue de litterature : c est l objet du chapitre 3.

- 2.1 Les modeles de langue et la tokenisation, avec le vocabulaire de 248 077 entrees qui
  reviendra a chaque chapitre
- 2.2 L alignement et ses deux etapes, SFT puis preferences, avec InstructGPT comme fil
- 2.3 DPO, explique a partir de sa fonction de perte, parce que la marge de recompense du
  chapitre 6 ne se comprend pas sans elle
- 2.4 LoRA et QLoRA, et pourquoi ils rendent possible ce qui ne l etait pas sur un T4
- 2.5 Le pre-entrainement continu : ce qu il fait et ne fait pas aux poids
- 2.6 Evaluer sans juge : log-vraisemblance, plancher, tests apparies

Figures : l illustration de LifeArchitect (empilement base puis alignement), notre schema
isometrique de la chaine a deux etapes, le redessin de LoRA d apres Hu et al., et deux
schemas a dessiner pour le scoring par log-vraisemblance et pour DPO.

---

## Chapitre 3 : Etat de l art

- 2.1 L alignement des modeles de langue : SFT, modeles de recompense, RLHF, et la lignee
  InstructGPT
- 2.2 DPO, et pourquoi il remplace RLHF ici : pas de modele de recompense a entrainer, donc
  un budget compatible avec deux T4
- 2.3 Le pre-entrainement continu pour les langues peu dotees, et le cas d AfriqueQwen
- 2.4 Evaluer sans juge : pourquoi un juge est inacceptable quand l equipe ne lit pas la
  langue cible, et ce que la log-vraisemblance permet a la place
- 2.5 Les jeux disponibles en haoussa, leurs licences et leur provenance
- 2.6 Ce qui manque dans la litterature, et que ce stage adresse

---

## Chapitre 4 : Methodologie

Le coeur scientifique du rapport.

- 4.1 La question, formulee comme une comparaison a une seule variable
- 4.2 Le dispositif a six etats : A0, A1, A2s, A3s, A2d, A3d, et les trois ecarts qu il
  permet de mesurer
- 4.3 Ce qui est tenu constant, et comment on le verifie plutot que de le supposer :
  empreintes de la partition d evaluation, identite des tokenizers, nombre de pas
- 4.4 Les donnees de chaque etape, et pourquoi elles conviennent a leur objectif
- 4.5 Les trois axes d evaluation, leurs metriques, leurs planchers
- 4.6 Les tests statistiques : McNemar pour les mesures appariees, bootstrap apparie pour le
  macro F1 qui n a pas de test analytique
- 4.7 La contamination decouverte dans Uhura, et la decision de n evaluer que sur 183
  questions

---

## Chapitre 5 : Mise en oeuvre

Chez Tchoumi, l implementation. Ici, la chaine technique.

- 5.1 L environnement : Kaggle, ses deux sessions GPU simultanees, son plafond de session
- 5.2 QLoRA : quantification en 4 bits, adaptateurs LoRA, et ce que cela permet sur un T4
- 5.3 La chaine, etape par etape, avec les entrees et les sorties de chacune
- 5.4 Les garde-fous contre les echecs silencieux, qui sont la vraie difficulte du projet :
  adaptateur pose sur le mauvais backbone, module reste en cache, dataset perime, graine de
  partition confondue avec la graine d entrainement
- 5.5 Les obstacles techniques et leur resolution : memoire insuffisante en SFT, surcout du
  critere d arret en evaluation, fragmentation memoire en DPO, tous ramenes a la meme cause

---

## Chapitre 6 : Resultats

- 6.1 Ce que le pre-entrainement continu apporte : modelisation de la langue, capacite
- 6.2 Ce qu il n apporte pas : la veracite
- 6.3 L alignement n amplifie pas l avantage, sur deux des trois axes
- 6.4 L exception : l interaction mesuree sur l axe Harmless
- 6.5 La replication sur trois graines, et ce qu elle valide ou invalide
- 6.6 Les limites, enoncees et chiffrees : puissance statistique, une seule langue, un axe
  sur trois, effet loge dans les classes minoritaires

---

## Conclusion

Ce qui est etabli, ce qui ne l est pas, et la consequence pratique pour qui envisage de
payer un pre-entrainement continu en esperant de l alignement. Perspectives : seconde
langue, axes de capacite supplementaires.

---

## Bibliographie

Format bibtex, cles `auteur_annee_motcle`.

---

## Ce qui alimente chaque chapitre, et qui existe deja

| chapitre | sources dans le depot |
| :---- | :---- |
| 2, 3 | `05_References/`, `06_Reading_Notes/` du vault |
| 4 | `03_Experiments/Plan_Evaluation.md`, `Decision_Graines.md` |
| 5 | `.claude/skills/pipeline/SKILL.md`, les notebooks, `src/` |
| 6 | `03_Experiments/Resultats_E4.md`, `Resultats_DPO.md`, les dix figures deja produites |

Les figures de `07_Presentations/figures/` sont directement reutilisables : elles ont ete
produites pour la soutenance et couvrent les courbes, les volumes de donnees, les resultats
et les intervalles.
