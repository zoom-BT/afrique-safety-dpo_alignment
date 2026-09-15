# Rapport de stage pre-ingenieur : conventions de travail

On redige en Markdown, on convertit en LaTeX a la fin. Ces conventions existent pour que la
conversion soit **mecanique** et non un travail de reprise.

## Arborescence

```
rapport_/
  00_frontmatter/     pages liminaires, une par fichier
  chapitres/          un fichier par chapitre, numerotes
  figures/            images, nommees selon la convention ci-dessous
  logos/              enspy.png, uy1.png, et le logo de la structure d accueil
  tableaux/           donnees sources des tableaux, en CSV
  bibliographie/      references.bib
```

## Nommer les figures

Le nom du fichier porte le numero de chapitre et un identifiant court.

```
figures/fig_03_dispositif_six_etats.png
figures/fig_05_interaction_afrihate.png
```

En Markdown, on ecrit :

```markdown
![Le dispositif a six etats, une seule variable](../figures/fig_03_dispositif_six_etats.png)
{#fig:03_dispositif_six_etats}
```

Ce qui devient, sans ambiguite :

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_03_dispositif_six_etats.png}
  \caption{Le dispositif a six etats, une seule variable}
  \label{fig:03_dispositif_six_etats}
\end{figure}
```

**La legende est la phrase entre crochets.** Elle doit se suffire a elle meme : un lecteur
qui feuillette le rapport doit comprendre la figure sans lire le paragraphe qui la precede.

## Nommer les tableaux

Meme principe, prefixe `tab`.

```markdown
Table: Volumes de donnees a chaque etape {#tab:03_volumes}
```

Les donnees sources vont dans `tableaux/tab_03_volumes.csv` quand le tableau depasse cinq
lignes, pour qu on puisse le regenerer sans le retaper.

## Renvoyer a une figure ou un tableau

Toujours par son identifiant, jamais par sa position.

```markdown
Le dispositif est represente en [@fig:03_dispositif_six_etats].
```

Ecrire "la figure ci-dessus" casse a la conversion, parce que LaTeX replace les flottants.

## Citer

Les cles bibtex suivent `auteur_annee_motcle`.

```markdown
Le protocole suit celui d InstructGPT [@ouyang_2022_instructgpt].
```

Le fichier `bibliographie/references.bib` est la seule source. Aucune reference ecrite a la
main dans le texte.

## Regles de redaction

**Jamais de tiret cadratin.** Ni le caractere directement, ni le `---` LaTeX qui le produit.
Utiliser deux points, un point virgule, une virgule, ou une parenthese.

**Les chiffres cites doivent etre verifiables.** Tout nombre du rapport vient d un fichier de
resultats, pas d une memoire. En cas de doute, le recalculer.

**Les nombres en francais** prennent la virgule decimale et l espace insecable comme
separateur de milliers : 0,0405 et 2 810.

**Une affirmation, une preuve.** Un resultat annonce sans son intervalle, son effectif ou son
test ne tient pas dans un rapport de recherche.

## Conversion vers LaTeX

Prevu avec pandoc, gabarit base sur `memoir` comme le rapport de reference. La commande
exacte sera fixee quand la structure sera stable, pour ne pas figer trop tot un choix de
gabarit.

Les outils LaTeX de la machine sont dans `F:\Open2Work` : MiKTeX, avec la configuration
`pdflatex` dans `.vscode/settings.json`.
