# Chapitre 5 : Resultats

Ce chapitre presente les mesures dans l ordre ou elles repondent a la question du stage :
ce que le pre-entrainement continu apporte avant tout alignement, ce qu il n apporte pas,
ce que l alignement change ou ne change pas a cet ecart, puis ce que la replication sur
trois graines valide. Il se termine par les limites, chiffrees.

Sauf mention contraire, les chiffres sont ceux de la graine 42, la premiere. Les deux autres
graines sont traitees en 5.5.

## 5.1 Ce que le pre-entrainement continu apporte

Avant de mesurer quoi que ce soit sur les axes d evaluation, la chaine d entrainement
elle-meme fournit une premiere mesure : la perte du SFT, c est-a-dire la log-vraisemblance
negative des demonstrations haoussa sous chaque modele. La figure [@fig:05_courbes_sft] en
donne les courbes.

![Courbes de perte du SFT pour les deux backbones, sur les memes donnees et la meme recette.
A gauche, les deux trajectoires ; a droite, la descente totale de
chacune.](../figures/fig_05_courbes_sft.png)
{#fig:05_courbes_sft}

Deux lectures s imposent, et la seconde est la plus importante.

**Le backbone pre-entraine part tres largement devant.** A la fin du SFT, la perte vaut
1,324 pour AfriqueQwen contre 2,222 pour Qwen-Base, soit un ecart de 0,90 nat. En termes de
perplexite, le modele pre-entraine est 2,5 fois plus a l aise sur ces phrases haoussa. Son
exactitude de prediction du token suivant est de 0,744 contre 0,642, dix points de plus.

**Mais les deux apprennent la meme chose.** La descente totale vaut 0,518 pour AfriqueQwen
et 0,567 pour Qwen-Base. L ecart entre les deux courbes est une translation constante d un
bout a l autre de l entrainement : le SFT ne le referme pas, et ne le creuse pas non plus.
Le pre-entrainement continu a donne un meilleur point de depart, pas un meilleur
apprentissage.

Ce constat se prolonge sur les axes de capacite. Le tableau [@tab:05_depart] donne l ecart
entre les deux backbones bruts, avant tout alignement, sur les trois axes.

Table: Ecart de depart entre les deux backbones bruts, A1 moins A0, graine 42
{#tab:05_depart}

| axe | A0 Qwen-Base | A1 AfriqueQwen | ecart | test |
| :---- | ---: | ---: | ---: | :---- |
| Helpful, exactitude | 0,168 | 0,300 | +0,132 | McNemar, p = 2,7e-5 |
| Harmless, exactitude | 0,335 | 0,562 | +0,227 | McNemar, p = 1e-27 |
| Harmless, macro F1 | 0,288 | 0,310 | +0,021 | bootstrap, contient zero |
| Honest, exactitude | 0,372 | 0,372 | 0,000 | McNemar, p = 1 |

Sur l arithmetique, treize points. Sur la classification de messages, vingt-deux points
d exactitude : sans pre-entrainement continu, Qwen-Base obtient 0,335 la ou repondre
"normal" a tout donnerait 0,567. Il fait pire que la reponse constante ; il est inutilisable
sur cette tache. Avec le pre-entrainement, le modele la traite.

## 5.2 Ce qu il n apporte pas : la veracite

La derniere ligne du tableau [@tab:05_depart] est le contraste central du rapport. Sur
l axe Honest, les deux backbones bruts obtiennent exactement le meme score, 0,3716, sur les
183 questions non contaminees. Tous deux sont au-dessus du hasard, a 0,263, avec une valeur p
de 0,0014 : ils comprennent la langue. Mais strictement au meme niveau.

Le detail est plus parlant que la moyenne. Les deux modeles sont en desaccord sur 30
questions, et ces desaccords se repartissent exactement 15 contre 15. Sur le jeu complet de
808 questions, mesure en debut de projet, ils divergent sur une reponse sur cinq. Le
pre-entrainement continu n a donc pas laisse le modele inchange : il a modifie son
comportement sur une part importante des questions. Mais les gains annulent exactement les
pertes.

L explication qui tient sur l ensemble des mesures est la suivante. Le pre-entrainement
continu achete la capacite a **traiter** le haoussa : classer, comprendre, calculer,
predire. La veracite n est pas une capacite de traitement. C est une question de ce que le
modele **croit**, et ces croyances viennent du pre-entrainement d origine. Lire 35 milliards
de tokens de texte africain ne corrige pas des idees recues sur le CERN ou sur l autobahn.

## 5.3 L alignement n amplifie pas l avantage

La question du stage porte sur l interaction entre backbone et alignement : l ecart entre
les deux bras change-t-il quand on les aligne ? La figure [@fig:05_six_etats_trois_axes]
donne les six etats sur les trois axes, et le tableau [@tab:05_ecarts] l ecart entre bras a
chaque etape.

![Les six etats de modele sur les trois axes. En rouge, le bras sans pre-entrainement
continu ; en vert, le bras avec. Le trait pointille marque le plancher de chaque
axe.](../figures/fig_05_six_etats_trois_axes.png)
{#fig:05_six_etats_trois_axes}

Table: Ecart entre bras, A3 moins A2, a chaque etape de l alignement, graine 42
{#tab:05_ecarts}

| axe, metrique | brut | apres SFT | apres SFT et DPO | tendance |
| :---- | ---: | ---: | ---: | :---- |
| Honest, exactitude | 0,000 | +0,011 | -0,016 | nul partout |
| Helpful, exactitude | +0,132 | +0,092 | +0,080 | persiste |
| Harmless, exactitude | +0,227 | +0,223 | +0,215 | stable |
| Harmless, macro F1 | +0,021 | +0,078 | +0,062 | s elargit |

Sur trois des quatre lignes, la reponse est negative.

**Sur Honest**, rien a aucune etape. L ecart final vaut moins 0,016 avec une valeur p de
0,71 : indiscernable de zero. Le backbone n apporte aucun benefice sur l axe sur lequel on
entraine.

**Sur Helpful**, l avantage du pre-entrainement persiste et reste significatif apres
alignement, avec une valeur p de 0,012. Il ne s amplifie pas.

**Sur Harmless en exactitude**, l ecart est enorme mais deja present avant tout
entrainement, et ne bouge pratiquement pas : c est une propriete du backbone, pas un effet
de l alignement.

Le SFT et le DPO ont pourtant bien fonctionne dans les deux bras. La figure
[@fig:05_courbes_dpo] montre les marges de recompense du DPO : elles partent de 0,017 et
finissent a 0,65 pour un bras, 0,56 pour l autre, soit un facteur quarante. Le signal etait
present dans les 709 paires, et les deux bras l ont extrait. Le resultat nul sur Honest
n est donc pas un echec d entrainement ; c est une absence d effet du backbone.

![Perte et marges de recompense du DPO pour les deux bras. Les marges sont mesurees contre
la reference gelee propre a chaque bras et ne se comparent pas entre
eux.](../figures/fig_05_courbes_dpo.png)
{#fig:05_courbes_dpo}

Une precaution de lecture s impose sur cette figure. La recompense implicite du DPO se
mesure contre le modele de reference gele **propre a chaque bras**. Une marge plus large
dit qu un bras s est davantage eloigne de son propre point de depart, pas qu il est mieux
aligne. La section 5.5 montrera d ailleurs que ces marges finales ne se reproduisent pas
d une graine a l autre. Elles servent ici a une seule chose : etablir que l apprentissage a
eu lieu.

## 5.4 L exception : l interaction sur l axe Harmless

La quatrieme ligne du tableau [@tab:05_ecarts] se comporte autrement. En macro F1 sur
AfriHate, l ecart entre bras part de 0,021, non significatif, et atteint 0,078 apres SFT
puis 0,062 apres DPO. Il triple. La figure [@fig:05_interaction] decompose ce mouvement.

![L interaction sur l axe Harmless. A gauche, le macro F1 de chaque bras avant et apres
alignement ; a droite, les trois quantites avec leur intervalle de confiance a 95 pour cent
par bootstrap apparie.](../figures/fig_05_interaction.png)
{#fig:05_interaction}

Les deux bras partent indiscernables : l intervalle de l ecart initial contient zero. Puis
l alignement les separe **dans des directions opposees**. Le bras sans pre-entrainement
perd 0,017 de macro F1, intervalle de moins 0,033 a moins 0,003. Le bras avec en gagne
0,023, intervalle de 0,006 a 0,044. Les deux effets excluent zero.

La quantite qui interesse le projet est la difference de ces deux differences :
l alignement a-t-il davantage aide un backbone que l autre ? Elle vaut 0,0405, avec un
intervalle de 0,0177 a 0,0664. Elle exclut zero.

C est le seul des trois axes ou l hypothese du stage se verifie. Et il se trouve que c est
l axe que l on n entraine pas : les 709 paires du DPO portent sur la veracite, pas sur la
moderation. L effet est donc un transfert entre axes, conditionne par le backbone. Aligner
un modele sans pre-entrainement continu sur la veracite degrade sa capacite de moderation ;
aligner le meme modele avec pre-entrainement l ameliore.

La figure [@fig:05_intervalles_afrihate] donne les six quantites testees sur cet axe, pour
que le lecteur voie lesquelles excluent zero et lesquelles ne l excluent pas, plutot que de
ne voir que celle qui arrange.

![Les six quantites testees sur l axe Harmless, chacune avec son intervalle a 95 pour cent.
Deux mille reechantillonnages apparies, les memes indices de lignes tires pour les six
etats.](../figures/fig_05_intervalles_afrihate.png)
{#fig:05_intervalles_afrihate}

Une reserve doit accompagner ce resultat, et la figure [@fig:05_f1_par_classe] la
localise. L effet porte sur le macro F1 et non sur l exactitude, ce qui signifie qu il se
loge dans les classes minoritaires : 351 messages abusifs et 103 haineux, contre 595
normaux. Pour un axe de securite, ce sont precisement les classes qui comptent. Ce sont
aussi celles ou le bruit d echantillonnage est le plus fort.

![F1 par classe sur AfriHate, pour les etats bruts et alignes des deux bras. L effet de
l alignement se concentre sur les classes abusif et
haineux.](../figures/fig_05_f1_par_classe.png)
{#fig:05_f1_par_classe}

## 5.5 La replication sur trois graines

Tous les chiffres qui precedent viennent d une seule graine. Deux autres graines, 43 et 44,
ont rejoue la chaine entiere a l identique, en ne changeant que l initialisation des
adaptateurs et l ordre des exemples. La graine de partition est restee fixee, comme
explique en 3.3.

**Le SFT se reproduit.** La figure [@fig:05_sft_trois_graines] superpose les six courbes,
et la figure [@fig:05_dispersion_graines] en donne la dispersion.

![Perte du SFT pour deux backbones et trois graines. Les trois courbes d un meme bras se
superposent presque.](../figures/fig_05_sft_trois_graines.png)
{#fig:05_sft_trois_graines}

![Dispersion de la perte finale du SFT sur trois graines, avec pour chaque bras sa propre
echelle. A droite, l ecart entre bras rapporte au bruit de
graine.](../figures/fig_05_dispersion_graines.png)
{#fig:05_dispersion_graines}

Table: Perte finale du SFT et marge finale du DPO sur trois graines, moyenne et ecart-type
{#tab:05_graines}

| quantite | A3 AfriqueQwen | A2 Qwen-Base | ecart entre bras |
| :---- | ---: | ---: | ---: |
| perte finale SFT | 1,344 +/- 0,020 | 2,246 +/- 0,049 | 0,902 +/- 0,055 |
| marge finale DPO | 0,591 +/- 0,104 | 0,578 +/- 0,035 | 0,013 |

L ecart entre backbones au SFT vaut vingt-six fois l ecart-type de graine. Le decalage de
0,9 nat n etait donc pas un accident d initialisation. C est le premier resultat du projet
dont on peut dire qu il est reproduit, et il l est largement.

**Les marges du DPO ne se reproduisent pas.** Sur la premiere graine, le bras pre-entraine
finissait avec une marge de 0,706 contre 0,585, et on aurait pu y lire un avantage. Sur
trois graines, les deux bras donnent 0,591 et 0,578, avec un ecart-type de 0,104 pour le
premier : l ecart de 0,013 entre eux vaut un dixieme du bruit. Ils sont indiscernables. La
figure [@fig:05_stabilite_metriques] met ce constat en regard des mesures sur donnees
jamais vues.

![Amplitude du changement entre graines pour cinq mesures du projet. Les metriques
d entrainement bougent, les mesures sur donnees tenues a l ecart ne bougent
pas.](../figures/fig_05_stabilite_metriques.png)
{#fig:05_stabilite_metriques}

La lecon depasse le cas present. Sur le meme dispositif, une metrique d entrainement s est
retournee d une graine a l autre pendant que les mesures sur donnees jamais vues se
reproduisaient a quelques millieres pres. Une metrique d entrainement decrit l ajustement
aux donnees vues ; seule une mesure tenue a l ecart peut porter une conclusion.

**Les conclusions tiennent.** La figure [@fig:05_replication_axes] compare l ecart entre
bras apres alignement sur les deux premieres graines, pour les trois axes.

![Ecart entre bras apres alignement, sur deux graines, pour les trois axes. Meme verdict
partout.](../figures/fig_05_replication_axes.png)
{#fig:05_replication_axes}

Sur Honest, chaque etat ne bouge que d une question sur 183 entre les graines, et l ecart
reste indiscernable de zero. Sur Helpful, l avantage reste significatif, et passe meme de
0,080 a 0,120 sur la seconde graine ; l erosion observee sur la premiere ne se reproduit
pas, et ne doit donc pas etre affirmee. Sur Harmless, l interaction vaut 0,0358 sur la
seconde graine, intervalle de 0,0146 a 0,0604, contre 0,0405 sur la premiere : les deux
intervalles excluent zero et se recouvrent largement. Les six quantites testees donnent le
meme verdict sur les deux graines.

La replication de la troisieme graine sur les axes d evaluation est en cours au moment de
la redaction ; ses resultats seront integres ici.

## 5.6 Les limites, chiffrees

Un resultat de recherche vaut par ce qu il exclut autant que par ce qu il affirme. Cinq
limites bornent ce qui precede.

**La puissance sur l axe Honest.** Avec 183 questions et une trentaine de desaccords, le
test de McNemar ne peut declarer significatif qu un ecart d au moins 7,1 points. L ecart
observe vaut 1,6 point. La formulation correcte est donc : on exclut un benefice important
du backbone sur la veracite, on n exclut pas un benefice petit. Toute affirmation plus forte
serait fausse.

**Une seule langue.** Tout le dispositif porte sur le haoussa. Le claim se lit "sur le
haoussa" et pas davantage. Une seconde langue, le yoruba, dispose de la meme chaine
d evaluation et de trois fois plus de donnees supervisees, et en constitue la suite
naturelle.

**Un axe sur trois.** Le resultat positif, l interaction, ne concerne que l axe Harmless.
Les deux autres axes sont nuls ou sans amplification.

**L effet dans les classes minoritaires.** L interaction porte sur le macro F1, donc sur
les classes abusif et haineux, 103 exemples pour la seconde. C est la ou un axe de securite
se joue, et c est aussi la ou le bruit est le plus fort.

**Le contenu des jeux.** Uhura est un TruthfulQA occidental, professionnellement traduit.
On mesure la veracite sur du savoir occidental exprime en haoussa, pas sur du savoir
africain. Et 89 pour cent des paires du DPO sont de meme origine. Un TruthfulQA de conception
africaine donnerait peut-etre un autre resultat ; il n existe pas, et c est en soi un constat
sur l ecosysteme.

Six quantites ont ete testees sur l axe Harmless. L interaction etait l hypothese de depart
du stage, formulee avant la mesure, ce qui n en fait pas une selection apres coup. Il faut
neanmoins le dire, parce qu un lecteur le verifiera.
