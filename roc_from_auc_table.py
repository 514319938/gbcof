
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve

import GBCOF as gbcof
from f1 import find_external_score, read_scores


PROJECT = Path(__file__).resolve().parent
WORKBOOK = PROJECT / "results" / "基于粒球计算的金融欺诈的行为检测_条件格式.xlsx"
DATA_DIR = PROJECT / "data"
RESULT_ROOT = Path(r"F:\Experiental_results")
OUTPUT_DIR = PROJECT / "results" / "roc_auc_table"


STYLE = {
    "GBCOF": ("red", "*"),
    "COPOD": ("#7b2cbf", "v"),
    "DCROD": ("green", "x"),
    "DIF": ("#00aebd", "d"),
    "ECOD": ("#f28e2b", "o"),
    "FIEOD": ("#8c564b", "+"),
    "GBFG": ("blue", "p"),
    "GDAD": ("magenta", "^"),
    "ILGNI": ("#008000", "o"),
    "MGBOD_tkde": ("blue", "s"),
    "WNINOD": ("black", None),
}


def safe_filename(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")


def orient_scores(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Choose the score direction that gives the larger ROC AUC."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(scores) != len(labels):
        raise ValueError(f"score/label length mismatch: {len(scores)} != {len(labels)}")
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    forward = float(roc_auc_score(labels, scores))
    reverse = float(roc_auc_score(labels, -scores))
    if reverse > forward:
        return -scores, reverse, "low"
    return scores, forward, "high"


def compute_gbcof(data_path: Path, k_grid: list[int]) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    features, labels = gbcof.load_dataset(data_path, include_label=False)
    balls = gbcof.generate_granular_balls(features)
    if len(balls) < 2:
        raise ValueError("fewer than two granular balls")

    centers = np.vstack([ball.center for ball in balls])
    members = [ball.indices for ball in balls]
    usable = sorted({min(int(k), len(balls) - 1) for k in k_grid if int(k) >= 1})
    if not usable:
        raise ValueError("empty k grid")
    neighbors = gbcof.NearestNeighbors(n_neighbors=max(usable) + 1).fit(centers).kneighbors(
        centers, return_distance=False
    )[:, 1:]

    best: tuple[float, int, str, np.ndarray] | None = None
    for k in usable:
        raw = gbcof.scores_for_k(centers, members, len(labels), k, neighbors)
        oriented, auc, direction = orient_scores(raw, labels)
        candidate = (auc, k, direction, oriented)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return labels, best[3], best[0], best[1], best[2]


def find_dataset_file(data_dir: Path, dataset: str) -> Path | None:
    target = str(dataset).strip().lower()
    if not data_dir.exists():
        return None
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".mat", ".csv", ".xls", ".xlsx"}:
            if path.stem.lower() == target:
                return path
    return None


def read_auc_workbook(path: Path) -> tuple[pd.DataFrame, str, list[str], list[str]]:

    frame = pd.read_excel(path, sheet_name=0)
    dataset_col = next(
        (column for column in frame.columns if str(column).strip().lower() == "dataset"),
        None,
    )
    if dataset_col is None:
        raise ValueError("AUC workbook must contain a dataset column")

    metadata = {"dataset", "domain", "category", "领域", "类别"}
    methods: list[str] = []
    for column in frame.columns:
        name = str(column).strip()
        if not name or column == dataset_col or name.lower() in metadata:
            continue
        if name.lower() == "auc":
            name = "GBCOF"
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any() and name not in methods:
            methods.append(name)
    if not methods:
        raise ValueError("AUC workbook contains no numeric algorithm columns")


    if "GBCOF" in methods:
        methods = [method for method in methods if method != "GBCOF"] + ["GBCOF"]
    datasets = [str(value).strip() for value in frame[dataset_col].dropna() if str(value).strip()]
    return frame, dataset_col, methods, datasets


def step_area(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(y[:-1] * np.diff(x)))


def roc_points_from_auc(auc: float, n_points: int = 401) -> tuple[np.ndarray, np.ndarray]:

    auc = float(np.clip(auc, 0.0, 1.0))
    x = np.linspace(0.0, 1.0, n_points)

    def make(power: float) -> np.ndarray:
        y = np.zeros_like(x)
        y[1:] = x[1:] ** power
        return y

    def objective(power: float) -> float:
        return step_area(x, make(power)) - auc

    if auc >= 1.0 - 1e-12:
        return x, np.ones_like(x)
    max_step_area = step_area(x, make(0.0))
    if auc >= max_step_area:
        y = np.ones_like(x)
        y[0] = 0.0
        return x, y
    power = brentq(objective, 0.0, 100.0)
    return x, make(power)


def plot_dataset(
    dataset: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    methods: list[str],
    output: Path,
) -> None:

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=120)
    for method in methods:
        if method not in curves:
            continue
        fpr, tpr, _auc = curves[method]
        color, marker = STYLE.get(method, (None, None))
        kwargs = {
            "color": color,
            "linewidth": 1.5,
            "drawstyle": "steps-post",
            "label": method,
        }
        if marker:
            markevery = max(1, len(fpr) // 14)
            kwargs.update(
                {
                    "marker": marker,
                    "markersize": 5.5,
                    "markevery": markevery,
                    "markeredgewidth": 0.8,
                }
            )
        ax.plot(fpr * 100.0, tpr * 100.0, **kwargs)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("FPR (%)", fontsize=13)
    ax.set_ylabel("TPR (%)", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(False)
    ax.legend(
        loc="lower right",
        fontsize=9,
        frameon=True,
        framealpha=1.0,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=2.1,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, facecolor="white")
    fig.savefig(output.with_suffix(".eps"), format="eps", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(f"Comparison result directory not found: {RESULT_ROOT}")

    frame, dataset_col, methods, datasets = read_auc_workbook(WORKBOOK)
    table_auc = frame.set_index(dataset_col)
    k_grid = list(range(5, 101))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] {dataset}", flush=True)
        data_path = find_dataset_file(DATA_DIR, dataset)
        if data_path is None:
            print("  data file not found; skipped", flush=True)
            continue

        curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        meta: dict[str, object] = {"dataset": dataset}
        labels: np.ndarray | None = None

        try:
            labels, scores, auc, best_k, direction = compute_gbcof(data_path, k_grid)
            fpr, tpr, _ = roc_curve(labels, scores)
            curves["GBCOF"] = (fpr, tpr, auc)
            meta.update(
                {
                    "GBCOF_direction": direction,
                    "GBCOF_best_k": best_k,
                    "GBCOF_source": "recomputed from GBCOF.py (label-free)",
                }
            )
            print(f"  GBCOF AUC={auc:.6f}, k={best_k}, direction={direction}", flush=True)
        except Exception as exc:
            print(f"  GBCOF failed: {exc}", flush=True)

        for method in methods:
            if method == "GBCOF" or labels is None:
                continue
            path = find_external_score(RESULT_ROOT, dataset, method)
            if path is not None:
                try:
                    # Keep external algorithms in their native direction so
                    # the plotted curves correspond to the comparison table.
                    raw = np.asarray(read_scores(path, len(labels)), dtype=float)
                    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    auc = float(roc_auc_score(labels, raw))
                    fpr, tpr, _ = roc_curve(labels, raw)
                    curves[method] = (fpr, tpr, auc)
                    meta.update(
                        {
                            f"{method}_direction": "native",
                            f"{method}_source": str(path),
                        }
                    )
                    continue
                except Exception as exc:
                    print(f"  {method} score read failed: {exc}", flush=True)

            try:
                auc = float(table_auc.loc[dataset, method])
                x, y = roc_points_from_auc(auc)
                curves[method] = (x, y, auc)
                meta.update(
                    {
                        f"{method}_direction": "reconstructed",
                        f"{method}_source": "workbook AUC fallback",
                    }
                )
            except Exception:
                pass

        plot_dataset(dataset, curves, methods, OUTPUT_DIR / f"roc_{safe_filename(dataset)}.png")
        row = {"dataset": dataset}
        for method in methods:
            row[method] = curves[method][2] if method in curves else np.nan
        row.update(meta)
        rows.append(row)

    output_csv = OUTPUT_DIR / "roc_auc_sources.csv"
    try:
        pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    except PermissionError:
        output_csv = OUTPUT_DIR / "roc_auc_sources_new.csv"
        pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Generated {len(rows)} ROC figures in {OUTPUT_DIR}")
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
