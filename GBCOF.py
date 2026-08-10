
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function


class GranularBall:

    def __init__(self, data: np.ndarray, indices: np.ndarray):
        self.data = data
        self.indices = indices
        self.n = len(data)
        self.center = np.mean(data, axis=0) if self.n else np.zeros(data.shape[1])
        distances = np.linalg.norm(data - self.center, axis=1) if self.n else np.zeros(0)
        self.radius = float(np.max(distances)) if self.n else 0.0

        self.dm = float(np.mean(distances)) if self.n else 0.0


def split_ball(ball: GranularBall, force: bool = False) -> list[GranularBall]:
    if ball.n <= 2:
        return [ball]
    points = ball.data
    p1 = points[np.argmax(np.linalg.norm(points - ball.center, axis=1))]
    p2 = points[np.argmax(np.linalg.norm(points - p1, axis=1))]
    if np.allclose(p1, p2):
        return [ball]
    d1 = np.linalg.norm(points - p1, axis=1)
    d2 = np.linalg.norm(points - p2, axis=1)
    left, right = d1 <= d2, d1 > d2
    if not left.any() or not right.any():
        return [ball]

    first = GranularBall(points[left], ball.indices[left])
    second = GranularBall(points[right], ball.indices[right])
    weighted_dm = (first.n * first.dm + second.n * second.dm) / ball.n
    if force or weighted_dm < ball.dm - 1e-6:

        return split_ball(first) + split_ball(second)
    return [ball]


def generate_granular_balls(X: np.ndarray) -> list[GranularBall]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    root = GranularBall(X, np.arange(len(X)))
    balls = split_ball(root)
    radii = np.asarray([ball.radius for ball in balls], dtype=float)
    threshold = 2.0 * max(float(radii.mean()), float(np.median(radii)))

    result: list[GranularBall] = []
    for ball in balls:
        if ball.radius > threshold:
            result.extend(split_ball(ball, force=True))
        else:
            result.append(ball)
    return result


@njit(cache=True)
def all_acd(centers: np.ndarray, neighbor_ids: np.ndarray) -> np.ndarray:
    m, k = neighbor_ids.shape
    dimensions = centers.shape[1]
    result = np.empty(m)
    for i in range(m):
        visited = np.zeros(k, dtype=np.bool_)
        best = np.empty(k)
        for j in range(k):
            distance = 0.0
            for col in range(dimensions):
                delta = centers[i, col] - centers[neighbor_ids[i, j], col]
                distance += delta * delta
            best[j] = np.sqrt(distance)

        total = 0.0
        for step in range(k):
            node, edge = -1, np.inf
            for j in range(k):
                if not visited[j] and best[j] < edge:
                    edge, node = best[j], j
            visited[node] = True
            total += 2.0 * (k - step) * edge / (k * (k + 1.0))
            for j in range(k):
                if not visited[j]:
                    distance = 0.0
                    for col in range(dimensions):
                        delta = centers[neighbor_ids[i, node], col] - centers[neighbor_ids[i, j], col]
                        distance += delta * delta
                    best[j] = min(best[j], np.sqrt(distance))
        result[i] = total
    return result


def scores_for_k(
    centers: np.ndarray,
    members: list[np.ndarray],
    n_samples: int,
    k: int,
    neighbors: np.ndarray,
) -> np.ndarray:
    ids = neighbors[:, :k]
    acd = all_acd(centers, ids)
    denominator = acd[ids].mean(axis=1)
    ball_scores = np.divide(acd, denominator, out=np.ones_like(acd), where=denominator > 0)
    scores = np.empty(n_samples, dtype=float)
    for value, indices in zip(ball_scores, members):
        scores[indices] = value
    return scores


def normalize_features(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    lower = np.min(matrix, axis=0)
    upper = np.max(matrix, axis=0)
    span = upper - lower
    normalized = np.divide(
        matrix - lower,
        span,
        out=np.zeros_like(matrix, dtype=float),
        where=span > 0,
    )
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def canonical(path: Path) -> str:
    name = path.stem.lower()
    if name.endswith("ori"):
        name = name[:-3]
    return name.rstrip("_")


def discover_files(data_dir: Path) -> list[Path]:
    priority = {".mat": 0, ".csv": 1, ".xlsx": 2, ".xls": 3}
    selected: dict[str, Path] = {}
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in priority:
            continue
        name = canonical(path)
        old = selected.get(name)
        if old is None or priority[path.suffix.lower()] < priority[old.suffix.lower()]:
            selected[name] = path
    return [selected[name] for name in sorted(selected)]


def read_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    content = loadmat(path)
    arrays = [
        value
        for key, value in content.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2
    ]
    if not arrays:
        raise ValueError("MAT file has no two-dimensional array")
    raw = np.asarray(max(arrays, key=lambda value: value.size), dtype=float)
    if raw.shape[1] < 2:
        raise ValueError("MAT data must contain at least one feature and one label column")
    return raw[:, :-1], raw[:, -1]


def looks_like_header(columns) -> bool:
    numeric = 0
    for column in columns:
        text = str(column).strip()
        numeric += bool(re.fullmatch(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", text))
    return numeric == 0


def read_table(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    reader = pd.read_excel if path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv
    try:
        frame = reader(path)
    except ImportError as exc:
        raise RuntimeError("legacy XLS input requires the xlrd package") from exc
    if not looks_like_header(frame.columns):
        frame = reader(path, header=None)
    frame = frame.replace({"?": np.nan, " ": np.nan, "": np.nan})
    if frame.shape[1] < 2:
        raise ValueError("tabular data must contain features and a final label column")
    return frame.iloc[:, :-1], frame.iloc[:, -1]


def load_dataset(path: Path, include_label: bool = False) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        features, labels = read_mat(path)
    else:
        table, labels = read_table(path)
        table = pd.get_dummies(table, dummy_na=True, dtype=float)
        features = SimpleImputer(strategy="most_frequent").fit_transform(table)

    labels = np.asarray(labels).reshape(-1)
    unique = np.unique(labels[~pd.isna(labels)])
    if len(unique) != 2:
        raise ValueError(f"non-binary labels ({len(unique)} classes)")
    y = (labels.astype(str) != str(unique.min())).astype(int)
    return normalize_features(features), y


__all__ = [
    "GranularBall",
    "NearestNeighbors",
    "all_acd",
    "canonical",
    "discover_files",
    "generate_granular_balls",
    "load_dataset",
    "normalize_features",
    "read_mat",
    "read_table",
    "scores_for_k",
    "split_ball",
]



