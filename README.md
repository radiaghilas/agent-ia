# RAG Médicaments

Ce projet implémente un système RAG (Retrieval-Augmented Generation) pour des notices médicales issues du BDPM (Base de Données Publique des Médicaments).

## Structure du projet

- `indexation.py` : construit un index vectoriel FAISS à partir des notices BDPM ou d'une API publique.
- `rag.py` : interroge l'index pour répondre à une question sur un médicament en utilisant Groq.
- `data/` : dossier contenant les sources de données (`CIS_RCP.csv` ou `CIS_RCP.zip`).
- `index_data/` : dossier généré contenant l'index FAISS et les métadonnées.

## Prérequis

- Python 3.10+ (Python 3.14 recommandé)
- Un environnement virtuel recommandé

## Installation

1. Créez et activez un environnement virtuel :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Installez les dépendances :

```powershell
pip install -r requirements.txt
```

> Si vous utilisez un GPU, installez la variante `faiss` adaptée à votre environnement.

## Configuration

1. Téléchargez le fichier CIS_RCP.zip depuis le site officiel :

https://www.data.gouv.fr/datasets/base-de-donnees-publique-des-medicaments-defi-idoc-sante

2. Placez le fichier `data/CIS_RCP.csv` ou `data/CIS_RCP.zip` dans le dossier `data/`.
3. Créez un fichier `.env` à la racine du projet :

```text
GROQ_API_KEY=VotreCleGroq
HF_TOKEN=VotreTokenHuggingFace   # optionnel, pour accélérer le téléchargement du modèle
```

## Construction de l'index

### Indexation locale à partir du BDPM

```powershell
python indexation.py --source local
```

### Reconstruire l'index même si un index existe déjà

```powershell
python indexation.py --source local --force
```

### Utiliser la source API publique si vous n'avez pas le fichier local

```powershell
python indexation.py --source api
```

### Source de secours minimale

```powershell
python indexation.py --source fallback
```

## Utilisation de `rag.py`

Une fois l'index créé, vous pouvez poser des questions :

```powershell
python rag.py --question "Quels sont les effets secondaires du Nurofen ?"
```

Options disponibles :

- `--top_k` : nombre de chunks à récupérer (par défaut `5`)
- `--threshold` : seuil de similarité minimal pour accepter une réponse (par défaut `0.1`)
- `--max_tokens` : budget de tokens pour la génération de réponse (par défaut `450`)

Exemple :

```powershell
python rag.py --question "Quels sont les effets indésirables du Doliprane ?" --top_k 6 --threshold 0.1 --max_tokens 500
```

## Comportement attendu

- `indexation.py` analyse les notices, nettoie le HTML et segmente le texte en chunks.
- `rag.py` récupère les chunks les plus pertinents, détecte le médicament nommé dans la question, puis génère une réponse en se basant explicitement sur les extraits.
- Si la requête ne correspond pas à une information fiable, le script indique qu'il ne trouve pas la réponse.

## Remarques

- Le modèle Hugging Face peut afficher un avertissement si `HF_TOKEN` n'est pas défini. Ce message est normal et ne bloque pas l'exécution.
- Si vous utilisez `--source local`, assurez-vous que `data/CIS_RCP.csv` ou `data/CIS_RCP.zip` est bien présent.
- Le fichier `index_data/config.json` conserve la liste des médicaments et les métadonnées de l'index.

## Fichiers clés

- `indexation.py` : génération de l'index vectoriel
- `rag.py` : requêtes sur l'index et génération de réponses
- `data/CIS_RCP.csv` ou `data/CIS_RCP.zip` : sources BDPM
- `index_data/` : index stocké et métadonnées
