# Chapitre 3 : Methodologie

Ce chapitre expose le dispositif experimental. Il part de la question posee, la traduit en
une comparaison a une seule variable, puis decrit ce qu il a fallu tenir constant, mesurer et
verifier pour que l ecart observe soit attribuable a cette variable et a rien d autre.

## 3.1 La question, ramenee a une comparaison

La question du stage tient en une phrase : l alignement d un modele de langue fonctionne-t-il
mieux quand il part d un backbone pre-entraine en continu sur des langues africaines que
quand il part de la base d origine de ce backbone ?

Pour y repondre, il faut deux modeles qui ne different que par ce point. AfriqueQwen3.5-4B
a ete obtenu par pre-entrainement continu de Qwen3.5-4B-Base sur 35,5 milliards de tokens en
cinquante langues africaines. Les deux partagent l architecture, la taille et, comme on le
verifiera en 3.3, le vocabulaire. Ils forment donc la paire ideale : la seule difference
entre eux est ce que le pre-entrainement continu a modifie dans les poids.

On applique aux deux la meme chaine d alignement, avec les memes donnees, la meme recette et
la meme graine. Si l un des deux s aligne mieux, le backbone est la seule explication
disponible.

## 3.2 Le dispositif a six etats

L alignement se fait en deux etapes, comme dans InstructGPT : un ajustement supervise sur
des demonstrations (SFT), puis une optimisation de preferences sur des paires (DPO). Chaque
backbone traverse les deux, ce qui donne six etats de modele, representes en
[@fig:03_dispositif_six_etats].

![Le dispositif a six etats. Deux backbones, trois niveaux d entrainement, et les trois
ecarts qu ils permettent de mesurer.](../figures/fig_03_dispositif_six_etats.png)
{#fig:03_dispositif_six_etats}

Table: Les six etats de modele et ce que chacun sert a mesurer
{#tab:03_etats}

| etat | backbone | entrainement | role |
| :---- | :---- | :---- | :---- |
| A0 | Qwen3.5-4B-Base | aucun | reference |
| A1 | AfriqueQwen-50Langs | aucun | ecart de depart |
| A2s | Qwen3.5-4B-Base | SFT | contribution du SFT seul |
| A3s | AfriqueQwen-50Langs | SFT | contribution du SFT seul |
| A2d | Qwen3.5-4B-Base | SFT puis DPO | la question du stage |
| A3d | AfriqueQwen-50Langs | SFT puis DPO | la question du stage |

Trois ecarts en decoulent, et il faut les trois.

**L ecart de depart**, A1 moins A0, mesure ce que le pre-entrainement continu apporte avant
tout alignement. Sans lui, un avantage observe a la fin ne pourrait pas etre distingue d un
avantage qui preexistait.

**L ecart apres SFT seul**, A3s moins A2s, isole la contribution de la premiere etape.

**L ecart final**, A3d moins A2d, repond a la question. Mais sans le point intermediaire, il
ne dirait pas laquelle des deux etapes a produit l effet. C est l ablation qu InstructGPT
pratique entre ses propres etapes, reproduite ici.

## 3.3 Ce qui est tenu constant, et comment on le verifie

Un dispositif a une seule variable ne vaut que si tout le reste est reellement identique.
Quatre elements ont ete verifies sur les fichiers eux-memes, plutot que supposes.

**La partition d evaluation.** Le decoupage des donnees entre entrainement et evaluation est
sauvegarde par chaque run. Les fichiers des deux bras ont la meme empreinte MD5 : les deux
backbones ont vu rigoureusement les memes exemples, dans le meme role.

**La tokenisation.** Le pre-entrainement continu aurait pu modifier le vocabulaire, ce qui
aurait rendu les sequences differentes entre bras. Les deux tokenizers ont ete compares sur
l integralite du jeu d evaluation : 248 077 entrees de vocabulaire, et 87 694 tokens produits,
identiques au token pres. Les deux bras voient donc exactement les memes sequences.

**Le nombre de pas.** Les deux bras ont fait 348 pas de SFT et 135 pas de DPO, avec la meme
memoire de crete a 0,03 Go pres. Rien dans la trajectoire d entrainement ne les distingue en
dehors de la perte elle-meme.

**Les deux graines.** La graine d entrainement fait varier l initialisation des adaptateurs
et l ordre des exemples. La graine de partition decide quels exemples vont en evaluation. Ce
sont deux variables distinctes dans le code, et la seconde vaut 42 pour toutes les
experiences, quelle que soit la premiere. Les faire varier ensemble melangerait variance
d entrainement et variance de decoupage, et un ecart entre graines ne mesurerait plus ce
qu on veut.

## 3.4 Les donnees de chaque etape

Chaque etape a un objectif different et demande des donnees de nature differente. La figure
[@fig:03_flux_donnees] donne les volumes, la figure [@fig:03_inventaire_jeux] la provenance
et les licences.

![Volumes de donnees a chaque etape, et ce qui a du etre ecarte. Les barres sont
normalisees : chaque etape occupe la meme largeur, les segments montrent les
proportions.](../figures/fig_03_flux_donnees.png)
{#fig:03_flux_donnees}

**Le SFT apprend un format.** Il doit faire passer un modele qui complete du texte a un
modele qui repond a une instruction. Il lui faut des demonstrations naturelles. Le jeu Aya
de Cohere en fournit 3 512 en haoussa, redigees par des locuteurs et non traduites, avec une
validation humaine documentee, sous licence Apache-2.0. Sur tous les criteres examines,
provenance, licence et validation, aucune source trouvee ne fait mieux. Le decoupage donne
2 810 exemples d entrainement et 702 tenus a l ecart.

**Le DPO apprend une preference.** Il lui faut des paires : une reponse preferee, une
reponse rejetee, a une meme question. Deux sources en fournissent sans qu aucun modele ait a
les fabriquer. Uhura-TruthfulQA publie pour chaque question une meilleure reponse et des
reponses incorrectes ; on prend la premiere de chaque. UbuntuGuard publie des transcriptions
deja etiquetees conformes ou non. Le socle compte 886 paires, 791 d Uhura et 95 d UbuntuGuard
sur l axe Honest, decoupees en 709 d entrainement et 177 tenues a l ecart, sans aucune
question commune aux deux cotes.

Une limite de ce socle doit etre dite : 89 pour cent du signal est du contenu occidental
traduit. Seules les 95 paires d UbuntuGuard sont ancrees en contexte africain. On aligne
donc la veracite en haoussa sur des idees recues americaines, ce qui est une limite du jeu et
non du protocole, et qui sera reprise au chapitre 5.

![Tous les jeux utilises, avec leur role, leur volume, leur licence et leur
provenance.](../figures/fig_03_inventaire_jeux.png)
{#fig:03_inventaire_jeux}

## 3.5 Les trois axes d evaluation

L evaluation suit la decomposition classique en trois axes, utile (helpful), honnete
(honest) et inoffensif (harmless). Elle repose sur un principe unique : **aucun modele
n evalue jamais le texte d un autre modele.**

La raison est pratique autant que methodologique. L equipe ne lit pas le haoussa. Un juge
automatique dont on ne peut pas etablir la fiabilite invaliderait toute la chaine, et un
juge humain n etait pas disponible. On compare donc partout des nombres, jamais des textes,
ce qui rend le scoring aussi fiable en haoussa qu en anglais.

Table: Les trois axes, leur jeu, leur metrique et leur plancher
{#tab:03_axes}

| axe | jeu | metrique | n | plancher |
| :---- | :---- | :---- | ---: | ---: |
| Honest | Uhura-TruthfulQA, QCM | log-vraisemblance des options | 183 | 0,263 |
| Helpful | AfriMGSM | correspondance de la reponse numerique | 250 | aucun |
| Harmless | AfriHate | macro F1 par log-vraisemblance | 1 049 | 0,241 |

**Sur l axe Honest**, le modele recoit une question et ses options ecrites. On calcule la
log-vraisemblance qu il attribue a chaque option, normalisee par sa longueur en tokens, et
l option la plus probable est sa reponse. La normalisation est indispensable : une somme
brute favorise mecaniquement les reponses courtes, alors que dans TruthfulQA la bonne
reponse est souvent la plus longue et la plus nuancee.

Le plancher est recalcule depuis le nombre reel d options par question et non suppose egal a
un quart : toutes les questions n en ont pas quatre. Il vaut 0,263 sur les 183 questions
retenues.

**Sur l axe Helpful**, le modele resout un probleme d arithmetique et sa reponse est un
nombre, compare au nombre attendu. Les modeles evalues etant des checkpoints de base, ils ne
repondent pas sans exemples : le protocole standard de MGSM, huit solutions detaillees dans
la langue cible, est applique a tous les etats. Cet axe n est pas entraine ; il sert a
detecter un oubli catastrophique, c est-a-dire un alignement qui ameliorerait la veracite en
detruisant une capacite.

**Sur l axe Harmless**, le modele classe un message en normal, abusif ou haineux. La
classification est scoree comme le QCM, par log-vraisemblance des trois mots d etiquette,
sans generation. Le macro F1 est prefere a l exactitude parce que le jeu est desequilibre :
595 messages normaux, 351 abusifs, 103 haineux. Repondre "normal" partout donnerait 56,7
pour cent d exactitude pour un macro F1 de seulement 0,241, qui est le plancher retenu.

Cet axe n est pas entraine non plus : UbuntuGuard ne fournit que 26 paires haoussa sur ce
theme, trop peu pour un entrainement. Il mesure donc un transfert entre axes : aligner sur la
veracite change-t-il quelque chose a la moderation ?

## 3.6 Les tests statistiques

Un score au-dessus de son plancher ne prouve rien par lui-meme, et un ecart entre deux
scores non plus. Deux tests sont utilises, chacun pour la question qu il sait trancher.

**Le test binomial bilateral** repond a la question : ce score depasse-t-il le hasard ? Il
somme la probabilite de tous les resultats au moins aussi improbables que celui observe, ce
qui est la construction correcte pour une distribution non symetrique.

**Le test de McNemar** repond a la question : ces deux etats different-ils ? Les six etats
repondent aux memes questions. Les traiter comme deux echantillons independants jetterait
cette information, et avec elle de la puissance statistique deja payee en heures de GPU. Le
test ne regarde que les questions ou les deux etats sont en desaccord : celles que les deux
reussissent ou ratent ne disent rien sur lequel est meilleur. Il est calcule de facon exacte,
parce que les desaccords sont peu nombreux et que l approximation par le khi-deux n est pas
fiable en dessous d une vingtaine.

Le cout de l ignorer a ete mesure. Sur les 808 questions d Uhura, A1 depassait A0 de 17
questions ; un test de deux proportions ne distingue pas cet ecart de zero, avec une valeur p
de 0,38, quand le test apparie donne 0,21. La conclusion ne change pas ici, mais elle aurait
pu.

**Le bootstrap apparie** sert pour le macro F1, qui n a pas de test analytique. La quantite
qui interesse le projet sur l axe Harmless est une difference de differences : l alignement
a-t-il davantage aide un backbone que l autre ? Aucune variance en forme fermee n existe
pour cela. On reechantillonne donc les lignes avec remise, deux mille fois, en tirant **les
memes indices pour les six etats** puisqu ils ont classe les memes lignes, et on recalcule le
macro F1 de chacun a chaque tirage. Reechantillonner chaque bras separement romprait la
correspondance et gonflerait l intervalle sans le signaler.

## 3.7 La contamination decouverte dans Uhura

Uhura-TruthfulQA publie deux configurations pour chaque langue : une en generation, avec
meilleure reponse et reponses incorrectes, et une en choix multiple. Ce sont deux mises en
forme **du meme TruthfulQA traduit**.

Or le DPO s entraine sur la premiere, et l evaluation de l axe Honest utilise la seconde.
Mesure : 785 des 808 questions du QCM apparaissent dans le jeu de generation, et 625 d entre
elles ont ete vues a l entrainement, soit 77,4 pour cent du jeu d evaluation.

Evaluer sur les 808 questions aurait donc mesure la memorisation, et l aurait mesuree **en
faveur des etats entraines**, c est-a-dire dans le sens du resultat espere. C est le pire
type d artefact : celui qui confirme ce qu on veut trouver.

La contamination a ete detectee avant le premier run d evaluation, en repondant a une
question sur la provenance des sources. La decision a ete de n evaluer que sur les 183
questions jamais vues : 160 venant de la partition d evaluation du DPO, et 23 absentes du jeu
de generation. Le notebook recalcule lui-meme le decoupage du DPO avec la meme graine, plutot
que de se fier a une liste ecrite ailleurs, et deux assertions verifient qu aucune question
d entrainement ne subsiste.

Le prix est la puissance statistique, et il faut le chiffrer plutot que le mentionner. Avec
183 questions et une trentaine de desaccords entre deux etats, le test de McNemar ne peut
declarer significatif qu un ecart d au moins 7,1 points, comme le montre la figure
[@fig:03_seuils_detection]. Un ecart plus petit, s il existe, restera invisible. Cette limite
est enoncee des maintenant parce qu elle conditionne la lecture des resultats de l axe Honest
au chapitre 5.

![Ecart observe contre plus petit ecart detectable, par axe. Un resultat non significatif
n informe que si la mesure pouvait voir un effet.](../figures/fig_03_seuils_detection.png)
{#fig:03_seuils_detection}

Les deux autres axes ne sont pas concernes : ni AfriMGSM ni AfriHate n ont servi a
l entrainement, et leurs effectifs sont conserves entiers.
