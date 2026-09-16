# Chapitre 5 : Mise en oeuvre

Ce chapitre decrit la chaine technique qui a produit les resultats du chapitre 5. Il suit
l ordre d execution reel : l environnement d abord, puis la methode d entrainement qu il
impose, puis chaque etape avec ses entrees et ses sorties. Il se termine par les garde-fous
et les obstacles, parce que c est la que le temps du stage a reellement ete depense.

## 5.1 L environnement de calcul

Le stage ne disposait d aucun GPU local. Tout l entrainement et toute l evaluation ont tourne
sur les offres gratuites de Kaggle et de Google Colab, ce qui a fixe trois contraintes des le
depart.

**La memoire.** Un T4 de Kaggle offre 14,56 Go de memoire video. Un modele de quatre
milliards de parametres en pleine precision en demanderait seize pour ses seuls poids, avant
tout gradient et tout etat d optimiseur. L entrainement complet est donc exclu, et c est ce
qui impose QLoRA, decrit en 5.2.

**Le temps.** Kaggle plafonne une session a neuf heures et le quota hebdomadaire a trente
heures de GPU. Un bras entier, SFT puis DPO, dure environ neuf heures : trop pour une seule
session. Chaque etape a donc ete decoupee en soumissions independantes, l etape suivante
recuperant le resultat de la precedente par un dataset Kaggle attache, decrit en 5.3.

**La concurrence.** Kaggle autorise exactement deux sessions GPU simultanees. Cette limite
n est pas documentee ; elle a ete mesuree en tentant une troisieme soumission, refusee avec
le message correspondant. Elle a ete exploitee systematiquement : les deux bras d une meme
etape ont toujours tourne en parallele, ce qui a divise par deux le temps de mur.

Colab a servi aux evaluations de reference du chapitre 4, puis a ete abandonne pour la suite.
Ses sessions expirent sans preavis, effacent leur disque et epuisent un quota que le service
ne chiffre pas. Une session perdue a coute environ soixante-dix minutes de mesures, et c est
ce qui a fixe la regle enoncee en 5.4 : le depot est la source, l executant ne detient rien.

Le tableau [@tab:05_environnement] resume ces contraintes.

Table: Contraintes de l environnement de calcul et leur consequence sur le dispositif
{#tab:05_environnement}

| contrainte | valeur mesuree | consequence |
| :---- | :---- | :---- |
| memoire video | 14,56 Go par T4 | QLoRA obligatoire |
| plafond de session | 9 h | une soumission par etape |
| quota hebdomadaire | 30 h | une graine complete par semaine |
| sessions simultanees | 2 | les deux bras en parallele |

## 5.2 QLoRA : entrainer un modele de quatre milliards de parametres sur un T4

QLoRA combine deux idees. Les poids du modele de base sont quantifies en quatre bits et
geles : ils ne recoivent aucun gradient et occupent le quart de leur taille en demi
precision. L entrainement ne touche que des matrices de faible rang, les adaptateurs LoRA,
inserees dans les projections d attention et du bloc MLP.

Les parametres retenus, identiques pour les deux bras et pour les deux etapes, sont donnes
dans le tableau [@tab:05_qlora].

Table: Configuration QLoRA commune a toutes les experiences
{#tab:05_qlora}

| parametre | valeur |
| :---- | :---- |
| quantification | 4 bits, NF4, double quantification |
| rang LoRA | 16 |
| alpha LoRA | 32 |
| abandon LoRA | 0,05 |
| modules cibles | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| precision de calcul | fp16 |

Deux choix meritent une justification.

**La precision de calcul est fp16 et non bf16.** Le T4 est une carte Turing, qui ne possede
pas d unites bf16. Demander bf16 ne provoque pourtant aucune erreur : les activations
remontent silencieusement en fp32 et la memoire double. Le code retrograde donc
automatiquement vers fp16 quand la carte l exige.

**Les modules cibles sont listes explicitement.** La bibliotheque PEFT ne connait pas de
correspondance par defaut pour l architecture Qwen3.5 et leve une exception plutot que de
deviner. Un nom de module qui ne correspondrait a rien attacherait un adaptateur a rien, et
l entrainement tournerait a effet nul sans le signaler. La liste a ete verifiee contre un
point de controle reel avant tout run.

Le resultat concret est un adaptateur de 42,5 Mo par bras et par etape, la ou le modele
complet pese huit gigaoctets. C est ce qui rend possible le transfert entre soumissions
Kaggle decrit ci-dessous.

## 5.3 La chaine, etape par etape

La chaine comporte quatre etapes d entrainement et trois d evaluation. Chacune est un
notebook autonome, genere par script depuis le depot plutot que tape a la main, pour qu il
reste reproductible. Le tableau [@tab:05_chaine] en donne les entrees, les sorties et les
durees mesurees.

Table: Les etapes de la chaine, avec leurs entrees, sorties et durees mesurees sur la graine 42
{#tab:05_chaine}

| etape | entree | sortie | duree |
| :---- | :---- | :---- | ---: |
| SFT, bras A3 | AfriqueQwen + 2 810 exemples Aya | adaptateur A3_sft | 5,1 h |
| SFT, bras A2 | Qwen-Base + 2 810 exemples Aya | adaptateur A2_sft | 4,7 h |
| DPO, bras A3 | AfriqueQwen + A3_sft + 709 paires | adaptateur A3_dpo | 3,5 h |
| DPO, bras A2 | Qwen-Base + A2_sft + 709 paires | adaptateur A2_dpo | 3,8 h |
| E4 Uhura | six etats, 183 questions | exactitude par etat | 2,0 h |
| E4 AfriMGSM | six etats, 250 questions | exactitude par etat | 4,4 h |
| E4 AfriHate | six etats, 1 049 lignes | macro F1 par etat | 2,5 h |

**Le transfert entre etapes** passe par un dataset Kaggle prive, republie a chaque nouvel
adaptateur. Le dossier de travail d une soumission appartient au kernel qui l a ecrit et
n est pas lisible depuis un autre ; le dataset est le seul canal. Deux autres datasets
portent le code et les donnees brutes, ce qui fait trois a attacher a chaque soumission.

**Le SFT** entraine sur les 2 810 demonstrations Aya en haoussa, deux epoques, lot effectif
de seize, taux d apprentissage de 2e-5. Il produit 348 pas dans les deux bras.

**Le DPO** repart de l adaptateur SFT, pose sur le backbone, et l entraine sur les 709 paires
de preference, trois epoques, lot effectif de seize, taux d apprentissage de 5e-6. Le modele
de reference gele qu exige la methode est l adaptateur SFT lui-meme, ce qui evite d en
gerer une seconde copie. Il produit 135 pas dans les deux bras.

**L evaluation** charge chaque etat, c est-a-dire chaque couple backbone et adaptateur, en
quatre bits comme a l entrainement, et lui applique les trois axes du chapitre 4. Elle
ecrit son resultat apres chaque etat, pas a la fin, pour qu une session coupee au cinquieme
garde les quatre premiers.

## 5.4 Les garde-fous contre les echecs silencieux

La vraie difficulte du projet n a pas ete de faire tourner l entrainement. Elle a ete de
s assurer que ce qui tournait etait bien ce qu on croyait. Plusieurs modes de defaillance
produisent un resultat plausible, chiffre, sans lever la moindre erreur. Chacun de ceux
listes ici a ete rencontre ou evite de justesse, et chacun est desormais garde par une
verification explicite.

**Un adaptateur pose sur le mauvais backbone.** LoRA ajoute des matrices aux couches du
modele sans verifier d ou elles viennent. Poser l adaptateur du bras A2 sur AfriqueQwen
produirait un modele qui repond, qui obtient un score, et dont le score ne voudrait rien
dire. Chaque `adapter_config.json` declare le modele sur lequel il a ete entraine, et chaque
notebook compare cette declaration au modele qu il charge avant de continuer.

**Un adaptateur charge gele.** Le DPO repart d un adaptateur existant. Charge sans
l indicateur qui l autorise a s entrainer, il reste fige, la perte bouge a peine, et le run
se termine normalement. Un test unitaire verrouille l indicateur, et la verification finale
compare les poids : sur la graine 42, les 256 tenseurs de l adaptateur ont bouge entre
l entree et la sortie du DPO, et les 256 tenseurs de la reference gelee sont restes
identiques a l adaptateur SFT, a zero pres.

**Un module reste en cache.** Python conserve un module deja importe. Apres un `git pull`
qui met le fichier a jour sur le disque, la cellule continue de tourner sur l ancienne
version. Chaque notebook recharge explicitement les modules du projet et verifie la
presence d une fonction ajoutee recemment, refusant de demarrer sur une version perimee.

**Un chemin qui n existe plus.** Kaggle ne monte pas tous les datasets a la meme profondeur,
et supprime le dossier de tete d une archive quand il est seul a la racine. Un chemin code
en dur a fait echouer trois soumissions avant que la recherche ne devienne recursive et par
nom de dossier.

**Deux graines confondues.** La graine d entrainement fait varier l initialisation des
adaptateurs et l ordre des exemples. La graine de partition decide quels exemples vont en
evaluation. Les faire varier ensemble melangerait deux sources de variance, et l ecart
entre graines ne mesurerait plus ce qu on veut. Elles sont donc deux variables distinctes,
la seconde fixee a 42 pour toutes les experiences. La partition d evaluation des deux bras a
ete verifiee identique par empreinte MD5.

**La justesse par question jetee.** Une premiere version de l evaluation ne conservait que
l exactitude agregee, jugeant le detail trop volumineux. C est precisement ce qui interdit
le test apparie de McNemar, seul capable de trancher un ecart de dix-sept questions sur
huit cent huit. Le detail est desormais conserve partout, et pour la classification, les
etiquettes predites le sont aussi, sans quoi le bootstrap du chapitre 4 serait impossible.

## 5.5 Les obstacles techniques, et leur cause commune

Trois pannes ont coute chacune un ou plusieurs runs. Elles ont pris des formes differentes,
et se sont revelees avoir la meme racine : le vocabulaire de Qwen3.5 compte 248 077 entrees,
et toute operation qui le parcourt est couteuse.

**Memoire insuffisante en SFT.** La perte d entropie croisee materialise, pour chaque
position de la sequence, un vecteur de la taille du vocabulaire. A 1 024 positions et en
fp32, cela represente environ un gigaoctet par sequence, avant le gradient. Deux runs ont
echoue avant que la perte ne soit calculee par morceaux, ce qui a ramene la memoire de
crete a 7,5 Go.

**Surcout en evaluation.** Le critere d arret de la generation, qui interrompt le modele des
que la reponse est complete, indexe la totalite du vocabulaire a sa construction : cinq
secondes mesurees. Passe a la bibliotheque de la facon documentee, il etait reconstruit a
chaque question, soit vingt et une minutes de surcout pur sur deux cent cinquante
questions, davantage que la generation elle-meme. Il est desormais construit une fois par
lot.

**Fragmentation en DPO.** Le premier run de DPO a echoue faute de memoire alors que
l allocation demandee, 1,84 Go, etait inferieure de peu a l espace libre augmente de
3,38 Go reserves mais inutilises. La cause est la distribution des longueurs : les paires
Uhura font 52 tokens de mediane, celles d UbuntuGuard entre 598 et 973, et l allocateur
alternait entre de tout petits blocs et des blocs de 1,8 Go. Un reglage de l allocateur
autorisant les segments a croitre a suffi, sans rien tronquer.

La regle retenue de ces trois episodes tient en une phrase : sur ce modele, rien qui
parcoure le vocabulaire ne doit se trouver dans une boucle.

Il faut ajouter a cette liste une erreur de mesure qui n a coute aucun run mais aurait pu
fausser une decision. Une estimation des longueurs de sequence du SFT, faite sur la
concatenation brute des champs, donnait un maximum de 64 tokens et suggerait de reduire la
longueur maximale a 128. Refaite avec le gabarit de conversation reellement applique, la
mesure donne une mediane de 59 tokens mais un maximum de 4 661, et une reduction a 128
aurait tronque 22,4 pour cent des exemples. La longueur de 1 024 retenue en tronque deja
2,6 pour cent. La lecon vaut d etre ecrite : mesurer ce que la bibliotheque voit, pas ce
qu on croit lui donner.
