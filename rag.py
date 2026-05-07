#!/usr/bin/env python3
import argparse
import json
import os
import unicodedata
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Chemins des fichiers d'index et des métadonnées.
INDEX_DIR = Path("index_data")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_FILE = INDEX_DIR / "metadata.json"
CONFIG_FILE = INDEX_DIR / "config.json"


def load_index():
    """Charge l'index FAISS et les métadonnées associées depuis le disque."""
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Index indisponible. Exécutez d'abord indexation.py --force ou vérifiez le dossier index_data/."
        )
    index = faiss.read_index(str(INDEX_FILE))
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def normalize_search_text(text: str) -> str:
    """Normalise une chaîne pour la recherche en supprimant les accents et en passant en minuscules."""
    text = (text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def load_medications() -> list[str]:
    """Charge la liste des médicaments connus depuis le fichier de configuration."""
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            config = json.load(f)
            return config.get("medicaments", [])
    return []


def detect_medication(question: str, medications: list[str]) -> str | None:
    """Détermine si la question mentionne un médicament connu."""
    normalized_question = normalize_search_text(question)
    for med in medications:
        if normalize_search_text(med) in normalized_question:
            return med
    return None


def count_medication_chunks(metadata: list[dict], medication: str) -> int:
    """Compte le nombre de chunks associés à un médicament donné."""
    normalized_med = normalize_search_text(medication)
    return sum(
        1
        for item in metadata
        if normalize_search_text(item.get("medicament", "")) == normalized_med
    )


def get_sample_medication_chunks(metadata: list[dict], medication: str, limit: int = 5) -> list[dict]:
    """Retourne un sous-ensemble de chunks pour un médicament (utile pour debug ou démonstration)."""
    normalized_med = normalize_search_text(medication)
    return [
        item
        for item in metadata
        if normalize_search_text(item.get("medicament", "")) == normalized_med
    ][:limit]


def encode_question(question: str, model):
    """Encode une question en un vecteur pour la recherche par similarité."""
    vector = model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vector)
    return vector


def truncate_text(text: str, max_words: int = 80) -> str:
    """Raccourcit un texte long pour l'inclure proprement dans le prompt."""
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ... [texte tronqué]"


def retrieve_top_chunks(question: str, index, metadata, model, top_k: int = 5):
    """Récupère les meilleurs chunks en fonction de la similarité entre question et index."""
    query_vector = encode_question(question, model)
    distances, ids = index.search(query_vector, top_k)
    retrieved = []
    for score, idx in zip(distances[0], ids[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        retrieved.append({"score": float(score), **item})
    return retrieved


def rerank_medication_chunks(question: str, chunks: list[dict], model, top_k: int = 5):
    """Rerank des chunks déjà filtrés pour un médicament spécifique."""
    if not chunks:
        return []
    texts = [chunk.get("text", "") for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vectors)
    index_local = faiss.IndexFlatIP(vectors.shape[1])
    index_local.add(vectors)
    query_vector = encode_question(question, model)
    distances, ids = index_local.search(query_vector, min(top_k, len(chunks)))
    reranked = []
    for score, idx in zip(distances[0], ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        item = chunks[idx]
        reranked.append({"score": float(score), **item})
    return reranked


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """Construit le prompt de conversation envoyé à Groq."""
    system_prompt = (
        "Vous êtes un assistant d'information médicale. Répondez uniquement en vous appuyant sur les extraits fournis. "
        "Citez toujours la source et le médicament associé. "
        "Ajoutez systématiquement : Ces informations ne remplacent pas l'avis d'un professionnel de santé. "
        "Ne répondez que si le contexte contient explicitement l'information demandée par la question. "
        "Si le contexte ne mentionne pas cette information, répondez honnêtement que l'information n'est pas disponible dans les extraits fournis. "
        "N'inventez aucune donnée médicale, effets secondaires ou posologie qui ne figurent pas dans le contexte."
    )

    context_lines = [
        "Voici les extraits de notices disponibles :",
    ]
    for i, chunk in enumerate(chunks, start=1):
        context_lines.append(
            f"Extrait {i} — Médicament : {chunk.get('medicament')} — Source : {chunk.get('source')}\n{truncate_text(chunk.get('text', ''), max_words=120)}"
        )
    context = "\n\n".join(context_lines)
    user_prompt = (
        f"Question : {question}\n\n"
        "Répondez en français, en citant le numéro d'extrait et le médicament de chaque information. "
        "Ne basez votre réponse que sur les extraits fournis et ne complétez pas avec d'autres informations."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context + "\n\n" + user_prompt},
    ]


def create_answer(
    question: str,
    top_chunks: list[dict],
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    max_tokens: int = 450,
) -> str:
    """Crée la réponse finale en appelant l'API Groq avec le prompt construit."""
    if not top_chunks:
        return (
            "Je ne trouve pas cette information dans ma base de connaissances sur les médicaments indexés. "
            "Ces informations ne remplacent pas l'avis d'un professionnel de santé."
        )
    messages = build_prompt(question, top_chunks)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return content.strip()


def load_api_key() -> str:
    """Charge la clé d'API Groq depuis le fichier .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La clé GROQ_API_KEY est manquante. Créez un fichier .env contenant GROQ_API_KEY=..."
        )
    return api_key


def main() -> None:
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(description="Interroger le RAG de médicaments.")
    parser.add_argument("--question", help="Question en langage naturel à poser au RAG")
    parser.add_argument("--top_k", type=int, default=5, help="Nombre de chunks à récupérer")
    parser.add_argument("--threshold", type=float, default=0.1, help="Seuil de similarité minimale pour accepter une réponse")
    parser.add_argument("--max_tokens", type=int, default=450, help="Budget max de tokens pour la génération de réponse")
    args = parser.parse_args()

    index, metadata = load_index()
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)

    question = args.question
    if not question:
        question = input("Question : ")

    medications = load_medications()
    requested_med = detect_medication(question, medications)

    # Si un médicament est détecté, on élargit la recherche initiale pour améliorer les chances de trouver des chunks spécifiques.
    top_k_search = args.top_k if requested_med is None else max(args.top_k, 20)
    top_chunks = retrieve_top_chunks(question, index, metadata, model, top_k=top_k_search)

    # Si le médicament est mentionné, on priorise les chunks qui correspondent au médicament détecté.
    if requested_med:
        filtered = [
            chunk
            for chunk in top_chunks
            if normalize_search_text(chunk.get("medicament", "")) == normalize_search_text(requested_med)
        ]
        if filtered:
            top_chunks = filtered[: args.top_k]
        else:
            total_med_chunks = count_medication_chunks(metadata, requested_med)
            if total_med_chunks:
                med_chunks = [
                    item
                    for item in metadata
                    if normalize_search_text(item.get("medicament", "")) == normalize_search_text(requested_med)
                ]
                top_chunks = rerank_medication_chunks(question, med_chunks, model, top_k=args.top_k)
                if not top_chunks:
                    top_chunks = []

    if not top_chunks or top_chunks[0]["score"] < args.threshold:
        print(
            "Je ne trouve pas cette information dans ma base de connaissances sur les médicaments indexés. "
            "Ces informations ne remplacent pas l'avis d'un professionnel de santé."
        )
        return

    api_key = load_api_key()
    answer = create_answer(question, top_chunks, api_key, max_tokens=args.max_tokens)
    print("\n---\nRéponse :\n")
    print(answer)
    print("\nSources récupérées :")
    for chunk in top_chunks:
        print(f"- Médicament : {chunk.get('medicament')} | Source : {chunk.get('source')} | score : {chunk.get('score'):.3f}")


if __name__ == "__main__":
    main()
