"""
guardrail_model.py
Definición del vectorizador híbrido TF-IDF (Word + Char N-Grams) exportable a JSON,
vectorizador puro en NumPy (`vectorize_text_numpy`) libre de scikit-learn para Raspberry Pi,
y arquitectura neuronal `DomainGuardrailNet` entrenada desde pesos iniciales aleatorios
(Kaiming Normal + Xavier Uniform) para el proyecto Nano-AI.

Topología DomainGuardrailNet:
    input_dim -> 32 (ReLU) -> 16 (ReLU) -> 1 (Sigmoid)

Autor: David Isaí Portillo López (R2 - Model Lead)
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def extract_hybrid_ngrams(
    text: str,
    word_ngram_range: Tuple[int, int] = (1, 2),
    char_ngram_range: Tuple[int, int] = (3, 4),
) -> List[str]:
    """
    Extrae n-gramas híbridos de palabras (`w:...`) y de caracteres dentro de límites de palabra (`c:...`)
    a partir de un texto en inglés. Los n-gramas de caracteres otorgan robustez frente a typos
    o variaciones morfológicas (ej. 'temp'/'temperature', 'anomal'/'anomaly'/'anomalies').
    """
    cleaned = text.lower().strip()
    words = TOKEN_PATTERN.findall(cleaned)
    features: List[str] = []

    # 1. Word n-grams
    w_min, w_max = word_ngram_range
    n_words = len(words)
    for n in range(w_min, w_max + 1):
        for i in range(n_words - n + 1):
            ngram = " ".join(words[i : i + n])
            features.append(f"w:{ngram}")

    # 2. Character word-boundary n-grams (char_wb)
    c_min, c_max = char_ngram_range
    for word in words:
        padded = f"<{word}>"
        length = len(padded)
        for n in range(c_min, c_max + 1):
            for i in range(length - n + 1):
                features.append(f"c:{padded[i : i + n]}")

    return features


def fit_tfidf_vocab(
    train_texts: Iterable[str],
    train_labels: Optional[np.ndarray] = None,
    max_features: int = 256,
    word_ngram_range: Tuple[int, int] = (1, 2),
    char_ngram_range: Tuple[int, int] = (3, 4),
) -> Dict[str, Any]:
    """
    Ajusta el vocabulario TF-IDF y los pesos IDF exclusivamente sobre el conjunto de entrenamiento
    (evitando cualquier data leakage del conjunto de validación/prueba).

    Fórmula IDF suavizada estándar:
        idf(t) = ln((1 + N_docs) / (1 + df(t))) + 1.0
    """
    docs_list = list(train_texts)
    n_docs = len(docs_list)
    if n_docs == 0:
        raise ValueError("La lista `train_texts` no puede estar vacía.")

    doc_freq: Counter[str] = Counter()
    class1_freq: Counter[str] = Counter()
    class0_freq: Counter[str] = Counter()

    for idx, text in enumerate(docs_list):
        unique_terms = set(
            extract_hybrid_ngrams(
                text,
                word_ngram_range=word_ngram_range,
                char_ngram_range=char_ngram_range,
            )
        )
        for term in unique_terms:
            doc_freq[term] += 1
            if train_labels is not None:
                if int(train_labels[idx]) == 1:
                    class1_freq[term] += 1
                else:
                    class0_freq[term] += 1

    # Seleccionar los `max_features` términos más informativos dentro del set de entrenamiento
    scored_terms: List[Tuple[float, str]] = []
    for term, df in doc_freq.items():
        # Priorizar n-gramas de palabras y términos con alto poder discriminativo en train
        is_word_bonus = 1.35 if term.startswith("w:") else 1.0
        if train_labels is not None:
            c1 = class1_freq[term]
            c0 = class0_freq[term]
            diff_score = abs(c1 - c0) + 0.35 * df
        else:
            diff_score = float(df)
        scored_terms.append((diff_score * is_word_bonus, term))

    # Orden determinista por puntaje descendente y orden lexicográfico en caso de empate
    scored_terms.sort(key=lambda item: (-item[0], item[1]))
    selected_terms = [term for _, term in scored_terms[:max_features]]

    vocab: Dict[str, int] = {}
    idf_values: List[float] = []
    for idx, term in enumerate(selected_terms):
        vocab[term] = idx
        df = doc_freq[term]
        idf = float(np.log((1.0 + n_docs) / (1.0 + df)) + 1.0)
        idf_values.append(round(idf, 8))

    return {
        "metadata": {
            "project": "Nano-AI",
            "module": "DomainGuardrailNet TF-IDF Vectorizer",
            "language": "en",
            "author": "David Isaí Portillo López (R2 - Model Lead)",
            "num_train_docs": n_docs,
            "sublinear_tf": True,
            "norm": "l2",
        },
        "input_dim": len(vocab),
        "word_ngram_range": list(word_ngram_range),
        "char_ngram_range": list(char_ngram_range),
        "vocabulary": vocab,
        "idf": idf_values,
    }


def save_tfidf_vocab(vocab_dict: Dict[str, Any], output_path: Path | str) -> None:
    """Guarda el diccionario de vocabulario y pesos IDF en formato JSON portable."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)


def load_tfidf_vocab(vocab_path: Path | str) -> Dict[str, Any]:
    """Carga el vocabulario TF-IDF y pesos IDF desde un archivo JSON."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def vectorize_text_numpy(
    text_list: Iterable[str],
    vocab_dict: Dict[str, Any],
) -> np.ndarray:
    """
    Transforma una lista de frases en inglés en una matriz TF-IDF normalizada con norma L2
    utilizando únicamente Python estándar y NumPy (sin requerir scikit-learn en la Raspberry Pi).

    Fórmula por documento d y término t:
        tf_sublinear(t, d) = 1 + ln(count(t, d))  si count(t, d) > 0
        tfidf(t, d)        = tf_sublinear(t, d) * idf(t)
        v_norm(d)          = tfidf(d) / max(||tfidf(d)||_2, 1e-12)
    """
    texts = list(text_list)
    vocab: Dict[str, int] = vocab_dict["vocabulary"]
    idf = np.asarray(vocab_dict["idf"], dtype=np.float32)
    input_dim = int(vocab_dict["input_dim"])
    word_range = tuple(vocab_dict.get("word_ngram_range", [1, 2]))
    char_range = tuple(vocab_dict.get("char_ngram_range", [3, 4]))
    use_sublinear = bool(vocab_dict.get("metadata", {}).get("sublinear_tf", True))

    matrix = np.zeros((len(texts), input_dim), dtype=np.float32)

    for row_idx, text in enumerate(texts):
        ngrams = extract_hybrid_ngrams(
            text,
            word_ngram_range=(int(word_range[0]), int(word_range[1])),
            char_ngram_range=(int(char_range[0]), int(char_range[1])),
        )
        counts = Counter(ngrams)
        for term, count in counts.items():
            col_idx = vocab.get(term)
            if col_idx is not None and count > 0:
                tf_val = 1.0 + float(np.log(count)) if use_sublinear else float(count)
                matrix[row_idx, col_idx] = tf_val * idf[col_idx]

        # Normalización L2 por fila
        row_norm = float(np.linalg.norm(matrix[row_idx], ord=2))
        if row_norm > 1e-12:
            matrix[row_idx] /= row_norm

    return matrix


class DomainGuardrailNet(nn.Module):
    """
    Clasificador Binario Guardrail de Dominio Térmico para Nano-AI.
    Topología compacta:
        input_dim -> 32 (ReLU) -> 16 (ReLU) -> 1 (Sigmoid)
    Inicialización explícita desde pesos aleatorios:
        - Kaiming Normal (He) en capas ocultas con activación ReLU.
        - Xavier Uniform (Glorot) en capa de salida con activación Sigmoid.
    """

    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.act2 = nn.ReLU()
        self.fc_out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_random_weights()

    def _init_random_weights(self) -> None:
        """Inicialización explícita de pesos iniciales aleatorios (Random Starting Weights)."""
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        prob = self.sigmoid(self.fc_out(x))
        return prob


def summarize_guardrail_footprint(model: nn.Module, model_name: str = "DomainGuardrailNet") -> Dict[str, Any]:
    """
    Calcula e imprime el número exacto de parámetros entrenables y el peso en memoria (KB)
    en precisión FP32 para el clasificador Guardrail.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_kb = (param_bytes + buffer_bytes) / 1024.0

    layer_breakdown = []
    for layer_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            l_params = sum(p.numel() for p in module.parameters())
            layer_breakdown.append(
                f"{layer_name}: Linear({module.in_features} -> {module.out_features}) [{l_params:,} params]"
            )

    print("-" * 72)
    print(f"  Resumen de Arquitectura : {model_name}")
    print(f"  Parámetros Entrenables  : {trainable_params:,} (Total: {total_params:,})")
    print(f"  Tamaño en Memoria (FP32): {total_kb:.4f} KB ({param_bytes + buffer_bytes:,} bytes)")
    print("  Desglose de Capas Lineales:")
    for item in layer_breakdown:
        print(f"    - {item}")
    print("-" * 72)

    return {
        "model_name": model_name,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "size_bytes": param_bytes + buffer_bytes,
        "size_kb": total_kb,
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    net = DomainGuardrailNet(input_dim=256)
    summarize_guardrail_footprint(net)
