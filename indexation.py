#!/usr/bin/env python3
import argparse
import html
import json
import os
import re
import unicodedata
import zipfile
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Liste de médicaments recherchés dans les données BDPM.
MEDICAMENTS_CORPUS = [
    "doliprane",
    "dafalgan",
    "efferalgan",
    "ibuprofène",
    "advil",
    "nurofen",
    "aspirin",
    "aspégic",
    "amoxicilline",
    "augmentin",
    "smecta",
    "imodium",
    "ventoline",
    "becotide",
    "oméprazole",
    "inexium",
    "metformine",
    "glucophage",
]

# Chemins de fichiers utilisés par le pipeline d'indexation.
DATA_DIR = Path("data")
INDEX_DIR = Path("index_data")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_FILE = INDEX_DIR / "metadata.json"
CONFIG_FILE = INDEX_DIR / "config.json"
TEXT_FILE = DATA_DIR / "CIS_RCP.csv"
ZIP_FILE = DATA_DIR / "CIS_RCP.zip"


def normalize_text(text: str) -> str:
    """Nettoie le texte en uniformisant les retours de ligne et espaces."""
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_search_text(text: str) -> str:
    """Normalise un texte pour la recherche en supprimant accents et casse."""
    text = normalize_text(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def clean_html(html_text: str) -> str:
    """Supprime les balises HTML et conserve uniquement le texte visible."""
    if not html_text:
        return ""
    text = html.unescape(str(html_text))
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(text)


def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> list[str]:
    """Sépare un texte long en morceaux superposés pour la recherche vectorielle."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start : start + max_words]
        chunks.append(" ".join(chunk).strip())
        start += max_words - overlap
    return chunks


def load_cis_dataframe() -> pd.DataFrame:
    """Charge le fichier BDPM local, en CSV ou ZIP, et retourne un DataFrame pandas."""
    def read_csv_with_header_guess(file_obj):
        first_line = file_obj.readline().decode("latin-1", errors="ignore")
        has_header = "code_cis" in first_line.lower() or "rcp_html" in first_line.lower()
        file_obj.seek(0)
        header = 0 if has_header else None
        return pd.read_csv(file_obj, sep="\t", encoding="latin-1", header=header, usecols=[0, 1], dtype=str)

    if ZIP_FILE.exists():
        with zipfile.ZipFile(ZIP_FILE, "r") as z:
            text_names = [name for name in z.namelist() if name.upper().endswith("CIS_RCP.CSV")]
            if not text_names:
                raise FileNotFoundError(f"Aucun fichier CIS_RCP.csv trouvé dans {ZIP_FILE}")
            with z.open(text_names[0]) as f:
                df = read_csv_with_header_guess(f)
    elif TEXT_FILE.exists():
        with TEXT_FILE.open("rb") as f:
            df = read_csv_with_header_guess(f)
    else:
        raise FileNotFoundError(
            "Aucun fichier de données local trouvé. Placez data/CIS_RCP.zip ou data/CIS_RCP.csv dans le dossier data/."
        )
    df = df.fillna("")
    return df


def row_to_text(row: pd.Series, med_name: str) -> str:
    """Transforme une ligne de BDPM en texte propre pour l'indexation."""
    code_cis = str(row.iloc[0]) if len(row) > 0 else ""
    html_content = str(row.iloc[1]) if len(row) > 1 else ""
    parts = [
        f"Médicament : {med_name}",
        f"Code CIS : {code_cis}",
        clean_html(html_content),
    ]
    return normalize_text("\n".join(part for part in parts if part))


def build_local_corpus() -> list[dict]:
    """Construit le corpus local à partir du fichier BDPM et découpe les notices en chunks."""
    print("[indexation] Chargement des données locales BDPM...")
    df = load_cis_dataframe()
    print(f"[indexation] {len(df)} lignes chargées")
    corpus = []

    # Prépare une version normalisée de chaque ligne pour détecter les médicaments.
    text_series = df.fillna("").astype(str).agg(" ".join, axis=1).apply(normalize_search_text)

    for med_name in MEDICAMENTS_CORPUS:
        med_lower = normalize_search_text(med_name)
        mask = text_series.str.contains(re.escape(med_lower), na=False)
        matched = df[mask]
        if matched.empty:
            print(f"[indexation] Aucune ligne trouvée pour '{med_name}'")
            continue
        for row_index, row in matched.iterrows():
            text = row_to_text(row, med_name)
            chunks = chunk_text(text)
            for chunk_id, chunk in enumerate(chunks, start=1):
                corpus.append(
                    {
                        "medicament": med_name,
                        "source": str(TEXT_FILE if TEXT_FILE.exists() else ZIP_FILE),
                        "row_index": int(row_index),
                        "chunk_id": chunk_id,
                        "text": chunk,
                    }
                )
    return corpus


def flatten_json(value, prefix="") -> list[str]:
    """Aplatie récursivement une structure JSON en liste de textes exploitables."""
    texts = []
    if isinstance(value, dict):
        for k, v in value.items():
            texts.extend(flatten_json(v, prefix=f"{prefix}{k}."))
    elif isinstance(value, list):
        for item in value:
            texts.extend(flatten_json(item, prefix=prefix))
    elif isinstance(value, str):
        normalized = normalize_text(value)
        if normalized:
            texts.append(normalized)
    elif value is not None:
        texts.append(str(value))
    return texts


def fetch_medicament_api(med_name: str) -> list[dict]:
    """Récupère les données publiques d'un médicament via l'API publique des médicaments."""
    import requests

    url = "https://api.medicaments.gouv.fr/1.0/medicaments"
    params = {"denomination": med_name}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "medicaments" in payload:
        return payload["medicaments"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Réponse API inattendue pour les médicaments")


def build_api_corpus() -> list[dict]:
    """Construit un corpus à partir de données renvoyées par l'API des médicaments."""
    print("[indexation] Récupération via l'API publique des médicaments...")
    corpus = []
    for med_name in MEDICAMENTS_CORPUS:
        try:
            records = fetch_medicament_api(med_name)
        except Exception as exc:
            print(f"[indexation] Impossible de récupérer {med_name} : {exc}")
            continue
        if not records:
            print(f"[indexation] Aucune donnée API pour {med_name}")
            continue
        for record_idx, record in enumerate(records, start=1):
            text_pieces = flatten_json(record)
            text = normalize_text("\n".join(text_pieces))
            if not text:
                continue
            chunks = chunk_text(text)
            for chunk_id, chunk in enumerate(chunks, start=1):
                corpus.append(
                    {
                        "medicament": med_name,
                        "source": f"api:{med_name}",
                        "record_index": record_idx,
                        "chunk_id": chunk_id,
                        "text": chunk,
                    }
                )
    return corpus


def build_fallback_corpus() -> list[dict]:
    """Crée un corpus minimal de secours si aucune source de données n'est disponible."""
    print("[indexation] Création d'un corpus local minimal de secours...")
    fallback_text = (
        "Cet assistant contient des informations générales sur les médicaments. "
        "Il ne remplace pas un avis médical."
    )
    corpus = []
    for med_name in MEDICAMENTS_CORPUS[:8]:
        corpus.append(
            {
                "medicament": med_name,
                "source": "fallback",
                "chunk_id": 1,
                "text": fallback_text + f" Médicament: {med_name}.",
            }
        )
    return corpus


def build_corpus(source: str) -> list[dict]:
    """Sélectionne la source de création du corpus : local, API ou fallback."""
    if source == "local":
        return build_local_corpus()
    if source == "api":
        return build_api_corpus()
    return build_fallback_corpus()


def encode_corpus(corpus: list[dict], model_name: str = "paraphrase-multilingual-mpnet-base-v2") -> tuple[np.ndarray, list[dict]]:
    """Encode les textes du corpus avec SentenceTransformer pour construire l'index vectoriel."""
    if not corpus:
        raise ValueError("Corpus vide, impossible de construire l'index.")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[indexation] Utilisation du device : {device}")
    model = SentenceTransformer(model_name, device=device)
    texts = [item["text"] for item in corpus]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors, corpus


def save_index(vectors: np.ndarray, metadata: list[dict]) -> None:
    """Enregistre l'index FAISS et les métadonnées sur le disque."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dimension": dim,
                "model": "paraphrase-multilingual-mpnet-base-v2",
                "source": os.getenv("INDEX_SOURCE", "unknown"),
                "medicaments": MEDICAMENTS_CORPUS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[indexation] Index enregistré dans {INDEX_DIR}")


def main() -> None:
    """Point d'entrée du script d'indexation."""
    parser = argparse.ArgumentParser(description="Créer la base vectorielle des notices médicinales.")
    parser.add_argument(
        "--source",
        choices=["local", "api", "fallback"],
        default="local",
        help="Source de données à utiliser pour l'indexation",
    )
    parser.add_argument("--force", action="store_true", help="Forcer la reconstruction de l'index")
    args = parser.parse_args()

    if INDEX_FILE.exists() and METADATA_FILE.exists() and not args.force:
        print("[indexation] Un index existe déjà. Utilisez --force pour le reconstruire.")
        return

    try:
        corpus = build_corpus(args.source)
    except FileNotFoundError as exc:
        print(f"[indexation] Erreur : {exc}")
        if args.source == "local":
            print("[indexation] Essayez avec --source api ou créez data/CIS_RCP.zip")
            return
        raise

    if not corpus:
        print("[indexation] Corpus vide. Vérifiez la source de données.")
        return

    print(f"[indexation] {len(corpus)} chunks générés")
    vectors, metadata = encode_corpus(corpus)
    save_index(vectors, metadata)


if __name__ == "__main__":
    main()
