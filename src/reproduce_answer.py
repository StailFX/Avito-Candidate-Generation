"""Воспроизведение answer.csv из трёх исходных parquet-файлов.

При первом запуске скачивается зафиксированная версия multilingual-e5-small.
Все вычисления выполняются локально. Промежуточные массивы кешируются
в artifacts/reproduce/, чтобы можно было продолжить прерванный запуск.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRanker
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer
from threadpoolctl import threadpool_limits

from retrieval import FEATURES, ROOT, Retriever, TextIndex, topk
from semantic_adapter import unit
from semantic_context import ContextSignals, EXTRA_FEATURES
from validate_answer import validate


BASE_MODEL = "intfloat/multilingual-e5-small"
BASE_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EXPECTED_SHA256 = "db6826cb284becd9a00de2c29452bdff30be42afad9744877352a0a374cdd6ce"

MODEL_DIR = ROOT / "models" / "multilingual-e5-small"
ADAPTER_DIR = ROOT / "models" / "query_encoder_adapter"
RANKER_PATH = ROOT / "models" / "final_ranker.cbm"
CACHE = ROOT / "artifacts" / "reproduce"

HISTORY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
    "item_id",
    "item_title_raw",
    "item_location_id",
    "item_microcat_id",
    "item_latitude",
    "item_longitude",
]


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_input_files() -> None:
    required = ["train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet"]
    missing = [name for name in required if not (ROOT / "data" / name).exists()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Положите в папку data файлы: {names}")


def ensure_base_model() -> None:
    if (MODEL_DIR / "model.safetensors").exists():
        return
    print("Скачиваю зафиксированную версию multilingual-e5-small…", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=BASE_MODEL,
        revision=BASE_REVISION,
        local_dir=MODEL_DIR,
    )


def load_encoder(device: str, *, fine_tuned: bool) -> SentenceTransformer:
    model = SentenceTransformer(str(MODEL_DIR), local_files_only=True, device=device)
    model.max_seq_length = 128
    if fine_tuned:
        state = {}
        for path in sorted(ADAPTER_DIR.glob("adapter-*.safetensors")):
            state.update(load_file(str(path), device="cpu"))
        if not state:
            raise FileNotFoundError("Не найдены части query_encoder_adapter")
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Лишние параметры адаптера: {result.unexpected_keys}")
    model.eval()
    if device != "cpu":
        model.half()
    return model


def text_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def encode_cached(
    model: SentenceTransformer,
    texts: list[str],
    path: Path,
    batch_size: int,
) -> np.ndarray:
    meta = path.with_suffix(".json")
    digest = text_digest(texts)
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info.get("texts_sha256") == digest:
            return np.load(path, mmap_mode="r")

    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(path, vectors)
    meta.write_text(
        json.dumps({"texts_sha256": digest, "rows": len(texts)}, ensure_ascii=False, indent=2)
    )
    return np.load(path, mmap_mode="r")


def ensure_text_index() -> TextIndex:
    path = ROOT / "artifacts" / "text_index.joblib"
    if not path.exists():
        print("Строю лексический индекс…", flush=True)
        TextIndex.build()
    index = joblib.load(path, mmap_mode="r")
    index.__class__ = TextIndex
    return index


def base_embeddings(device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    items = pd.read_parquet(
        ROOT / "data" / "benchmark_items.parquet",
        columns=["item_title_raw", "item_description_raw"],
    )
    queries = pd.read_parquet(ROOT / "data" / "benchmark_queries.parquet")
    passages = (
        "passage: "
        + items.item_title_raw.fillna("")
        + ". "
        + items.item_description_raw.fillna("").str.replace(r"\\n", " ", regex=True).str[:900]
    ).tolist()
    query_texts = (
        "query: "
        + queries.search_query.fillna("")
        + ". "
        + queries.search_infm_params_text.fillna("")
    ).tolist()

    model = load_encoder(device, fine_tuned=False)
    item_vectors = encode_cached(model, passages, CACHE / "items_e5.npy", batch_size)
    query_vectors = encode_cached(model, query_texts, CACHE / "queries_e5.npy", batch_size)
    del model, items
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return item_vectors, query_vectors


def fine_tuned_queries(queries: pd.DataFrame, device: str, batch_size: int) -> np.ndarray:
    texts = (
        "query: "
        + queries.search_query.fillna("")
        + ". "
        + queries.search_infm_params_text.fillna("")
    ).tolist()
    model = load_encoder(device, fine_tuned=True)
    vectors = encode_cached(model, texts, CACHE / "queries_finetuned_e5.npy", batch_size)
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return vectors


def make_answer(output: Path, device: str, batch_size: int) -> None:
    ensure_input_files()
    ensure_base_model()
    CACHE.mkdir(parents=True, exist_ok=True)

    queries = pd.read_parquet(ROOT / "data" / "benchmark_queries.parquet")
    history = pd.read_parquet(ROOT / "data" / "train.parquet", columns=HISTORY_COLUMNS)
    index = ensure_text_index()
    item_vectors, old_queries = base_embeddings(device, batch_size)
    new_queries = fine_tuned_queries(queries, device, batch_size)

    docs = unit(item_vectors)
    retriever = Retriever(index, history, item_vectors, profile="current")
    signals = ContextSignals(index, history, queries, CACHE / "semantic_context")
    ranker = CatBoostRanker()
    ranker.load_model(str(RANKER_PATH))
    expected_features = FEATURES + EXTRA_FEATURES + [
        "finetuned_cos",
        "finetuned_geo",
        "finetuned_logrank",
        "finetuned_geo_logrank",
    ]
    if ranker.feature_names_ != expected_features:
        raise RuntimeError("Список признаков не совпадает с обученным ранжировщиком")

    answers: list[str] = []
    for i, row in enumerate(queries.itertuples(index=False)):
        with threadpool_limits(limits=4):
            dense_score = docs @ new_queries[i]
            same, probability, distance = retriever.history.geo(int(row.search_location_id))
            geo = 0.65 * same + 0.25 * np.sqrt(probability) + 0.10 * np.exp(-distance / 35)
            additional = np.unique(
                np.r_[topk(dense_score + 0.1 * geo, 200), topk(dense_score, 100)]
            )
            candidate_ids, base_features, _ = retriever.retrieve(
                row,
                old_queries[i],
                extra_candidates=additional,
            )
            context_features = signals.extra(i, candidate_ids, base_features)
            fine_score = docs[candidate_ids] @ new_queries[i]
            fine_geo = fine_score + 0.1 * geo[candidate_ids]
            fine_features = np.column_stack(
                [
                    fine_score,
                    fine_geo,
                    np.log1p(rankdata(-fine_score, method="average")),
                    np.log1p(rankdata(-fine_geo, method="average")),
                ]
            ).astype(np.float32)
            features = np.column_stack([base_features, context_features, fine_features])
            scores = ranker.predict(features, thread_count=4)
        answers.append(" ".join(index.ids[candidate_ids[topk(scores, 50)]]))
        if (i + 1) % 250 == 0:
            print(f"Готово запросов: {i + 1}/{len(queries)}", flush=True)

    result = pd.DataFrame({"query_id": queries.query_id.astype(str), "answer": answers})
    result.to_csv(output, index=False, encoding="utf-8")
    contract = validate(output, ROOT / "data")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    print(f"SHA256: {digest}")
    if digest == EXPECTED_SHA256:
        print("Файл совпадает с отправленным answer.csv")
    else:
        print("Формат корректен, но побайтовый хэш отличается. Проверьте версии библиотек и устройство.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Воспроизвести итоговый answer.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    make_answer(args.output, choose_device(args.device), args.batch_size)


if __name__ == "__main__":
    main()
