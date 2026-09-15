---
name: colab
description: Comment connecter un agent a Google Colab via le serveur MCP local, sur ce PC. A lire avant toute tentative d execution de cellule, et quand open_colab_browser_connection renvoie false.
---

# Connecter un agent a Colab, depuis ce PC

## L architecture, qui explique tout le reste

Le serveur MCP **n ouvre pas le navigateur**. Il demarre un serveur WebSocket **local** et
attend que le navigateur vienne a lui.

```
colab-mcp.exe  ---->  ws://localhost:<port aleatoire>
                              ^
                              |  connexion entrante
                       extension du navigateur
                              |
                        onglet Colab ouvert
```

Trois consequences, et elles sont la cause de presque toutes les pannes :

1. Le port est **choisi au hasard a chaque demarrage** du serveur. Redemarrer le serveur
   invalide la connexion en cours.
2. C est **l extension** qui se connecte. Sans extension active, rien ne se passe, et
   l appel renvoie simplement `false` sans expliquer pourquoi.
3. Le navigateur doit avoir **un onglet Colab ouvert** et charge.

## Ce qui est installe sur ce PC

| | |
| :---- | :---- |
| binaire | `C:\Users\Tchoutzine\.local\bin\colab-mcp.exe` |
| configuration | `~/.claude.json`, cle `mcpServers.colab-mcp` |
| arguments | aucun, seuls `-l` (dossier de logs) et `-p` (proxy) existent |
| delai | 120000 ms |
| journaux | `%TEMP%\colab-mcp-logs-<aleatoire>\colab-mcp.<date>.log` |

**Il n existe aucun parametre** pour choisir un compte Google, un profil de navigateur ou un
notebook. Le serveur trouve la session tout seul, ou ne la trouve pas.

## La procedure, dans cet ordre

L ordre compte, parce que le port change a chaque demarrage du serveur.

1. **Extension active dans le navigateur.** Fenetre normale de preference. En navigation
   privee, Chrome et Edge desactivent les extensions par defaut: il faut aller dans
   `chrome://extensions` ou `edge://extensions`, ouvrir les details de l extension ColabMCP,
   et activer l autorisation en navigation privee.
2. **Un seul compte Google connecte dans ce navigateur**, si le compte importe. L index
   `authuser` compte les comptes dans leur ordre de connexion et n est pas stable: avec un
   seul compte, l ambiguite disparait.
3. **Ouvrir le notebook Colab** et attendre qu il soit charge.
4. **Demarrer ou redemarrer le serveur MCP** si necessaire, avec `/mcp` dans le CLI.
5. **Recharger l onglet Colab**, pour que l extension se raccroche au nouveau port.
6. Appeler `open_colab_browser_connection`. Il doit renvoyer `true`.

## Diagnostiquer quand la connexion renvoie false

Lire le journal le plus recent. Trois lignes suffisent a situer le probleme.

```bash
recent=$(ls -t "$TEMP"/colab-mcp-logs-*/*.log | head -1)
grep -i "websocket server\|connection" "$recent" | tail -5
```

| ce que dit le journal | ce que cela signifie |
| :---- | :---- |
| aucune ligne `Starting WebSocket server` | le serveur MCP ne tourne pas |
| `Starting WebSocket server` mais rien apres | le serveur vit, aucun navigateur ne s est connecte: extension absente, desactivee, ou onglet Colab ferme |
| `connection open` puis `connection closed` | le navigateur s est connecte puis a lache: onglet ferme, passage en navigation privee, ou mise en veille |

La date du fichier de log dit aussi si le serveur a redemarre depuis la derniere tentative.
S il a redemarre, le port a change et il faut recharger l onglet.

## Pieges constates

**La navigation privee ne marche pas par defaut.** Chrome et Edge n y chargent pas les
extensions. Symptome: `open_colab_browser_connection` renvoie `false` alors que tout semble
en place.

**Changer de navigateur impose de reinstaller l extension.** Rien ne suit automatiquement.
Les navigateurs a base de Chromium (Chrome, Edge, Brave, Opera) acceptent les extensions du
Chrome Web Store. Firefox et Safari utilisent un moteur d extensions different.

**Les secrets Colab ne suivent pas non plus.** Sur un nouveau compte Google il faut recreer
`GITHUB_TOKEN` (le depot est prive) et `HF_TOKEN`, dans la cle du panneau de gauche, avec
l acces active pour le notebook.

**Une cellule ajoutee pendant que la connexion etait coupee peut ne jamais s executer.**
L appel revient vide, sans erreur, et `execution_count` reste a `null`. Recharger l onglet,
ou recreer la cellule.

**Un notebook ouvert depuis `colab.research.google.com/github/...` est une copie de
lecture.** Executer ses cellules fonctionne, mais rien n est reecrit vers GitHub. C est le
comportement voulu ici, le depot restant la source.

**Une session Colab peut etre reinitialisee sans prevenir.** Tout code ecrit directement
dans le navigateur disparait alors. Les notebooks doivent venir du depot, jamais l inverse.

## Verifier que la bonne session est vue

Apres connexion, lire les premieres cellules et comparer au notebook attendu. Le serveur se
connecte a **une** session navigateur, pas a un notebook nomme: si plusieurs onglets Colab
sont ouverts, il peut viser le mauvais. Fermer les autres leve l ambiguite.
