
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.metrics import precision_recall_curve

import GBCOF as gbcof


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TABLE = PROJECT_ROOT / "results" / "基于粒球计算的金融欺诈的行为检测_条件格式.xlsx"
DEFAULT_DATA = PROJECT_ROOT / "data"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_COMPARISON = Path(r"F:\Experiental_results")

METHOD_ORDER = [
    "GBCOF",
    "DCROD",
    "DIF",
    "GBDO",
    "GBFG",
    "GBMOD",
    "HGBAD",
    "ILGNI",
    "MFGAD",
    "MGBOD_tkde",
    "WFRDA",
]


def evaluation_best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return F1-Max over all score thresholds for a fixed score vector."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(scores) != len(labels):
        raise ValueError(f"score/label length mismatch: {len(scores)} != {len(labels)}")
    if len(np.unique(labels)) != 2:
        raise ValueError("labels must contain exactly two classes")
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    denominator = precision + recall
    values = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return float(np.max(values))


def best_f1_both_directions(scores: np.ndarray, labels: np.ndarray) -> tuple[float, str]:
    high = evaluation_best_f1(scores, labels)
    low = evaluation_best_f1(-np.asarray(scores), labels)
    if high >= low:
        return high, "high"
    return low, "low"


def parse_grid(text: str) -> list[int]:
    if ":" in text:
        start, stop = (int(value) for value in text.split(":", 1))
        if start < 1 or stop < start:
            raise ValueError("k-grid must be START:STOP")
        return list(range(start, stop + 1))
    values = sorted({int(value) for value in text.split(",") if value.strip()})
    if not values or values[0] < 1:
        raise ValueError("k-grid must contain positive integers")
    return values


def find_dataset_file(data_dir: Path, dataset: str) -> Path | None:
    target = dataset.lower()
    if not data_dir.exists():
        return None
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".mat" and path.stem.lower() == target:
            return path
    return None


def _find_child(parent: Path, name: str) -> Path | None:
    if not parent.exists():
        return None
    target = name.lower()
    for child in parent.iterdir():
        if child.name.lower() == target:
            return child
    return None


def _extract_zipped_score(method_root: Path, dataset: str, method: str) -> Path | None:
    expected = f"{dataset}_{method}".lower()
    for archive in method_root.iterdir():
        if not archive.is_file() or archive.suffix.lower() != ".zip":
            continue
        try:
            with zipfile.ZipFile(archive) as bundle:
                member = next(
                    (
                        name
                        for name in bundle.namelist()
                        if Path(name).suffix.lower() in {".mat", ".xls", ".xlsx"}
                        and Path(name).stem.lower() == expected
                    ),
                    None,
                )
                if member is None:
                    continue
                cache_dir = Path(tempfile.gettempdir()) / "gbcof_external_scores" / method
                cache_dir.mkdir(parents=True, exist_ok=True)
                destination = cache_dir / Path(member).name
                if not destination.exists():
                    destination.write_bytes(bundle.read(member))
                return destination
        except (OSError, zipfile.BadZipFile):
            continue
    return None


def find_external_score(result_root: Path, dataset: str, method: str) -> Path | None:
    method_root = _find_child(result_root, f"{method}_results")
    if method_root is None:
        return None

    dataset_dir = _find_child(method_root, dataset)
    if dataset_dir is None:
        for child in method_root.iterdir():
            if child.is_dir():
                dataset_dir = _find_child(child, dataset)
                if dataset_dir is not None:
                    break
    if dataset_dir is None:
        return _extract_zipped_score(method_root, dataset, method)

    if method == "MGBOD_tkde":
        expected = f"{dataset}_{method}".lower()
        for path in dataset_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".mat" and path.stem.lower() == expected:
                return path
        default = dataset_dir / "results_default.mat"
        if default.exists():
            return default

    expected = f"{dataset}_{method}".lower()
    for path in dataset_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".mat", ".xls", ".xlsx"}:
            if path.stem.lower() == expected:
                return path
    return None


def _vector_candidates(value, expected_length: int) -> list[tuple[int, np.ndarray]]:
    array = np.asarray(value)
    candidates: list[tuple[int, np.ndarray]] = []
    if array.ndim == 1 and array.size == expected_length:
        candidates.append((0, array))
    elif array.ndim == 2 and array.shape[0] == expected_length:
        candidates.extend((index, array[:, index]) for index in range(array.shape[1]))
    elif array.ndim == 2 and array.shape[1] == expected_length:
        candidates.extend((index, array[index, :]) for index in range(array.shape[0]))

    output: list[tuple[int, np.ndarray]] = []
    for index, candidate in candidates:
        try:
            vector = np.asarray(candidate, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if len(vector) == expected_length and np.isfinite(vector).any():
            output.append((index, vector))
    return output


def _is_binary_vector(vector: np.ndarray) -> bool:
    finite = vector[np.isfinite(vector)]
    if finite.size == 0:
        return True
    unique = np.unique(finite)
    return unique.size <= 2 and np.all(np.isin(unique, [0.0, 1.0]))


def read_mat_scores(path: Path, expected_length: int) -> np.ndarray:
    content = scio.loadmat(path)
    candidates: list[tuple[int, int, int, int, np.ndarray]] = []
    preferred = ("score", "outlier", "anomaly", "result", "value", "od")
    for key, value in content.items():
        if key.startswith("__"):
            continue
        key_rank = 0 if any(token in key.lower() for token in preferred) else 1
        for column_index, vector in _vector_candidates(value, expected_length):
            finite = vector[np.isfinite(vector)]
            distinct = int(np.unique(finite).size)
            binary_rank = 1 if _is_binary_vector(vector) else 0
            candidates.append((key_rank, binary_rank, -distinct, column_index, vector))
    if not candidates:
        raise ValueError(f"no numeric score vector of length {expected_length} in {path.name}")
    candidates.sort(key=lambda item: item[:4])
    return np.nan_to_num(candidates[0][4], nan=0.0, posinf=0.0, neginf=0.0)


def read_table_scores(path: Path, expected_length: int) -> np.ndarray:
    for header in (0, None):
        frame = pd.read_excel(path, header=header)
        for column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values) == expected_length:
                return values.to_numpy(dtype=float)
    raise ValueError(f"no numeric score column of length {expected_length} in {path.name}")


def read_scores(path: Path, expected_length: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return read_mat_scores(path, expected_length)
    if suffix in {".xls", ".xlsx"}:
        return read_table_scores(path, expected_length)
    raise ValueError(f"unsupported result file: {path}")


def load_workbook_spec(path: Path) -> tuple[list[str], list[str], pd.DataFrame]:
    frame = pd.read_excel(path, sheet_name=0)
    dataset_col = next((col for col in frame.columns if str(col).strip().lower() == "dataset"), None)
    if dataset_col is None:
        raise ValueError("AUC workbook must contain a dataset column")

    metadata_names = {
        "dataset",
        "domain",
        "category",
        "领域",
        "类别",
        "数据集",
    }
    methods: list[str] = []
    for column in frame.columns:
        name = str(column).strip()
        if not name or column == dataset_col or name.lower() in metadata_names:
            continue

        if name.lower() == "auc":
            name = "GBCOF"
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        if name in METHOD_ORDER or numeric_values.notna().any():
            methods.append(name)
    if not methods:
        raise ValueError("AUC workbook contains none of the expected algorithm columns")
    datasets = [str(value).strip() for value in frame[dataset_col].dropna() if str(value).strip()]
    return datasets, methods, frame


def prepare_gbcof(data_path: Path, include_label: bool, k_values: list[int]):
    X, labels = gbcof.load_dataset(data_path, include_label=include_label)
    balls = gbcof.generate_granular_balls(X)
    if len(balls) < 2:
        raise ValueError("fewer than two granular balls")
    centers = np.vstack([ball.center for ball in balls])
    members = [ball.indices for ball in balls]
    usable = sorted({min(int(k), len(balls) - 1) for k in k_values if int(k) >= 1})
    if not usable:
        raise ValueError("no usable k value")
    neighbors = gbcof.NearestNeighbors(n_neighbors=max(usable) + 1).fit(centers).kneighbors(
        centers, return_distance=False
    )[:, 1:]
    return labels, centers, members, neighbors, usable, len(balls)


def compute_gbcof_f1(
    data_path: Path,
    k_values: list[int],
    include_label: bool,
) -> tuple[float, int, str, int]:
    labels, centers, members, neighbors, usable, balls = prepare_gbcof(data_path, include_label, k_values)

    best: tuple[float, int, str] | None = None
    for k in usable:
        raw = gbcof.scores_for_k(centers, members, len(labels), k, neighbors)
        value, direction = best_f1_both_directions(raw, labels)
        candidate = (value, k, direction)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ValueError("no usable k value")
    return best[0], best[1], best[2], balls


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 evaluation for the AUC workbook")
    parser.add_argument("--auc-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--k-grid", default="5:100")
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--include-label",
        dest="include_label",
        action="store_true",
        default=False,
        help="include the target column in GBCOF distance features",
    )
    feature_group.add_argument(
        "--exclude-label",
        dest="include_label",
        action="store_false",
        help="exclude the target column from GBCOF features (default)",
    )
    parser.add_argument("--external-direction", choices=("best", "native"), default="best")
    args = parser.parse_args()

    datasets, methods, _ = load_workbook_spec(args.auc_table)
    k_values = parse_grid(args.k_grid)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] {dataset}", flush=True)
        data_path = find_dataset_file(args.data_dir, dataset)
        row: dict[str, object] = {"dataset": dataset}
        detail: dict[str, object] = {
            "dataset": dataset,
            "feature_mode": "paper-compatible" if args.include_label else "strict-label-free",
            "normalization": "min-max",
        }
        if data_path is None:
            detail["status"] = "data-not-found"
            for method in methods:
                row[method] = np.nan
            rows.append(row)
            details.append(detail)
            continue

        try:
            labels = gbcof.load_dataset(data_path, include_label=False)[1]
            value, best_k, direction, balls = compute_gbcof_f1(
                data_path,
                k_values,
                args.include_label,
            )
            row["GBCOF"] = value
            detail.update({"GBCOF_f1": value, "GBCOF_k": best_k, "GBCOF_direction": direction, "GBCOF_balls": balls})
            print(f"  GBCOF F1={value:.4f}, k={best_k}, direction={direction}, balls={balls}", flush=True)
        except Exception as exc:
            row["GBCOF"] = np.nan
            detail["GBCOF_error"] = str(exc)
            labels = None

        for method in methods:
            if method == "GBCOF":
                continue
            if labels is None:
                row[method] = np.nan
                continue
            score_path = find_external_score(args.comparison_root, dataset, method)
            if score_path is None:
                row[method] = np.nan
                detail[f"{method}_status"] = "score-not-found"
                continue
            try:
                scores = read_scores(score_path, len(labels))
                if args.external_direction == "best":
                    value, direction = best_f1_both_directions(scores, labels)
                else:
                    value, direction = evaluation_best_f1(scores, labels), "native"
                row[method] = value
                detail.update({f"{method}_f1": value, f"{method}_direction": direction, f"{method}_source": str(score_path)})
            except Exception as exc:
                row[method] = np.nan
                detail[f"{method}_error"] = str(exc)

        rows.append(row)
        details.append(detail)

    output = pd.DataFrame(rows, columns=["dataset", *methods])
    output_path = args.output_dir / "all_F1_results_new.xlsx"
    try:
        output.to_excel(output_path, index=False)
    except PermissionError:
        output_path = args.output_dir / "all_F1_results_new_.xlsx"
        output.to_excel(output_path, index=False)
    details_path = args.output_dir / "f1_details_new.csv"
    try:
        pd.DataFrame(details).to_csv(details_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        details_path = args.output_dir / "f1_details_new_.csv"
        pd.DataFrame(details).to_csv(details_path, index=False, encoding="utf-8-sig")
    print(f"Saved {output_path}")
    print(f"Saved {details_path}")


if __name__ == "__main__":
    main()
