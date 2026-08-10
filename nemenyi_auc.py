
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy.stats import friedmanchisquare, rankdata, studentized_range
from sklearn.metrics import roc_auc_score

import GBCOF as gbcof


PROJECT = Path(__file__).resolve().parent
WORKBOOK = PROJECT / "results" / "基于粒球计算的金融欺诈的行为检测_条件格式.xlsx"
DATA_DIR = PROJECT / "data"
OUTPUT_DIR = PROJECT / "results" / "nemenyi_auc"


def parse_grid(text: str) -> list[int]:
    if ":" in text:
        start, stop = (int(v) for v in text.split(":", 1))
        if start < 1 or stop < start:
            raise ValueError("k-grid must be START:STOP")
        return list(range(start, stop + 1))
    values = sorted({int(v) for v in text.split(",") if v.strip()})
    if not values or values[0] < 1:
        raise ValueError("k-grid must contain positive integers")
    return values


def best_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, str]:
    forward = float(roc_auc_score(labels, scores))
    reverse = float(roc_auc_score(labels, -scores))
    if forward >= reverse:
        return forward, "high"
    return reverse, "low"


def compute_gbcof_auc(data_path: Path, k_values: list[int]) -> tuple[float, int, str]:
    X, labels = gbcof.load_dataset(data_path, include_label=False)
    balls = gbcof.generate_granular_balls(X)
    if len(balls) < 2:
        raise ValueError("fewer than two granular balls")
    centers = np.vstack([ball.center for ball in balls])
    members = [ball.indices for ball in balls]
    usable = sorted({min(k, len(balls) - 1) for k in k_values})
    neighbors = gbcof.NearestNeighbors(n_neighbors=max(usable) + 1).fit(centers).kneighbors(
        centers, return_distance=False
    )[:, 1:]
    best: tuple[float, int, str] | None = None
    for k in usable:
        scores = gbcof.scores_for_k(centers, members, len(labels), k, neighbors)
        auc, direction = best_auc(scores, labels)
        candidate = (auc, k, direction)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ValueError("no usable k value")
    return best


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

    metadata_names = {"dataset", "domain", "category", "领域", "类别"}
    methods: list[str] = []
    for column in frame.columns:
        name = str(column).strip()
        if not name or column == dataset_col or name.lower() in metadata_names:
            continue
        if name.lower() == "auc":
            name = "GBCOF"
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any() and name not in methods:
            methods.append(name)
    if not methods:
        raise ValueError("AUC workbook contains no numeric algorithm columns")
    datasets = [str(value).strip() for value in frame[dataset_col].dropna() if str(value).strip()]
    return frame, dataset_col, methods, datasets


def compute_auc_matrix(data_dir: Path, workbook: Path, k_values: list[int]):

    frame, dataset_col, methods, datasets = read_auc_workbook(workbook)
    table = frame.set_index(dataset_col)
    rows: list[dict[str, float | str]] = []
    metadata: list[dict[str, object]] = []

    for dataset in datasets:
        data_path = find_dataset_file(data_dir, dataset)
        if data_path is None:
            print(f"skip {dataset}: data file not found")
            continue
        try:
            row: dict[str, float | str] = {"dataset": dataset}
            for method in methods:
                value = pd.to_numeric(table.loc[dataset, method], errors="coerce")
                if pd.isna(value):
                    raise ValueError(f"missing {method} AUC in workbook")
                row[method] = float(value)

            if "GBCOF" in methods:
                gb_auc, gb_k, gb_direction = compute_gbcof_auc(data_path, k_values)
                row["GBCOF"] = gb_auc
                metadata.append(
                    {
                        "dataset": dataset,
                        "GBCOF_best_k": gb_k,
                        "GBCOF_direction": gb_direction,
                        "GBCOF_feature_mode": "strict-label-free",
                        "GBCOF_source": "recomputed from GBCOF.py",
                    }
                )
            rows.append(row)
        except Exception as exc:
            print(f"skip {dataset}: {exc}")

    if not rows:
        raise RuntimeError("no complete dataset has scores for all algorithms")
    return pd.DataFrame(rows, columns=["dataset", *methods]), pd.DataFrame(metadata), methods


def nemenyi_test(auc_matrix: pd.DataFrame, alpha: float):
    methods = list(auc_matrix.columns[1:])
    values = auc_matrix[methods].to_numpy(dtype=float)
    ranks = np.vstack([rankdata(-row, method="average") for row in values])
    average_ranks = ranks.mean(axis=0)
    n_datasets, n_methods = ranks.shape

    friedman = friedmanchisquare(*[ranks[:, i] for i in range(n_methods)])

    q_critical = float(studentized_range.ppf(1.0 - alpha, n_methods, np.inf) / np.sqrt(2.0))
    standard_error = np.sqrt(n_methods * (n_methods + 1.0) / (6.0 * n_datasets))
    critical_difference = q_critical * standard_error

    pvalues = np.ones((n_methods, n_methods), dtype=float)
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            q_stat = abs(average_ranks[i] - average_ranks[j]) / standard_error
            p = float(studentized_range.sf(q_stat * np.sqrt(2.0), n_methods, np.inf))
            pvalues[i, j] = pvalues[j, i] = p

    rank_table = pd.DataFrame(ranks, columns=methods)
    rank_table.insert(0, "dataset", auc_matrix["dataset"].to_numpy())
    rank_table.loc[len(rank_table), "dataset"] = "Average rank"
    rank_table.loc[len(rank_table) - 1, methods] = average_ranks
    pvalue_table = pd.DataFrame(pvalues, index=methods, columns=methods)
    summary = pd.DataFrame(
        [
            {
                "n_datasets": n_datasets,
                "n_methods": n_methods,
                "alpha": alpha,
                "friedman_statistic": float(friedman.statistic),
                "friedman_pvalue": float(friedman.pvalue),
                "q_critical": q_critical,
                "critical_difference": critical_difference,
                "best_average_rank": methods[int(np.argmin(average_ranks))],
            }
        ]
    )
    average_rank_table = pd.DataFrame({"method": methods, "average_rank": average_ranks}).sort_values("average_rank")
    return rank_table, pvalue_table, summary, average_rank_table, critical_difference


def maximal_nonsignificant_groups(average_rank_table: pd.DataFrame, cd: float):
    ordered = average_rank_table.sort_values("average_rank").reset_index(drop=True)
    ranks = ordered["average_rank"].to_numpy(float)
    candidates: list[tuple[int, int]] = []
    for start in range(len(ordered)):
        end = start
        while end + 1 < len(ordered) and ranks[end + 1] - ranks[start] <= cd + 1e-12:
            end += 1
        if end > start:
            candidates.append((start, end))

    groups = [
        group
        for group in candidates
        if not any(
            other != group and other[0] <= group[0] and other[1] >= group[1]
            for other in candidates
        )
    ]
    return ordered, groups


def draw_cd_diagram(average_rank_table: pd.DataFrame, cd: float, summary: pd.DataFrame, output: Path):
    ordered, groups = maximal_nonsignificant_groups(average_rank_table, cd)
    k = len(ordered)
    blue = "#1018d7"
    red = "#ff1717"
    axis_y = 0.65
    left_x = 0.35
    right_x = k + 0.65
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.set_xlim(-1.35, k + 1.45)


    median_rank = (k + 1) / 2.0
    left = ordered[ordered["average_rank"] > median_rank].sort_values("average_rank", ascending=False).reset_index(drop=True)
    right = ordered[ordered["average_rank"] <= median_rank].sort_values("average_rank").reset_index(drop=True)
    label_step = 0.58
    label_top = 0.15
    left_y = [label_top - i * label_step for i in range(len(left))]
    right_y = [label_top - i * label_step for i in range(len(right))]
    bottom = min(left_y + right_y + [-0.8]) - 0.6
    top = 2.15 + 0.20 * max(1, len(groups))
    ax.set_ylim(bottom, top)

    def x_for_rank(rank: float) -> float:
        # Rank k is at the left and rank 1 is at the right.
        return k + 1.0 - rank

    # Top rank axis with descending labels: k, ..., 1.
    ax.plot([1, k], [axis_y, axis_y], color="black", linewidth=2.0)
    for rank in range(1, k + 1):
        x = x_for_rank(rank)
        ax.plot([x, x], [axis_y - 0.07, axis_y + 0.07], color="black", linewidth=1.7)
        ax.text(x, axis_y + 0.16, str(rank), ha="center", va="bottom", fontsize=15)


    cd_left = 0.95
    cd_right = cd_left + cd
    cd_y = top - 0.20
    ax.plot([cd_left, cd_right], [cd_y, cd_y], color=red, linewidth=2.4)
    ax.plot([cd_left, cd_left], [cd_y - 0.035, cd_y + 0.035], color=red, linewidth=1.2)
    ax.plot([cd_right, cd_right], [cd_y - 0.035, cd_y + 0.035], color=red, linewidth=1.2)
    ax.text((cd_left + cd_right) / 2, cd_y + 0.14, f"CD={cd:.3f}", color=red, ha="center", va="bottom", fontsize=17)


    group_y = axis_y - 0.30
    for level, (start, end) in enumerate(groups):
        y = group_y - level * 0.16
        x_left = x_for_rank(float(ordered.loc[end, "average_rank"]))
        x_right = x_for_rank(float(ordered.loc[start, "average_rank"]))
        ax.plot([x_left, x_right], [y, y], color=red, linewidth=2.0)
        ax.plot([x_left, x_left], [y - 0.045, y + 0.045], color=red, linewidth=1.0)
        ax.plot([x_right, x_right], [y - 0.045, y + 0.045], color=red, linewidth=1.0)


    for (_, item), y in zip(left.iterrows(), left_y):
        rank = float(item["average_rank"])
        method = str(item["method"])
        x = x_for_rank(rank)
        ax.plot([x, x], [axis_y - 0.04, y], color=blue, linewidth=1.6)
        ax.plot([x, left_x], [y, y], color=blue, linewidth=1.6)
        ax.text(left_x - 0.10, y, method, color=blue, ha="right", va="center", fontsize=13)
    for (_, item), y in zip(right.iterrows(), right_y):
        rank = float(item["average_rank"])
        method = str(item["method"])
        x = x_for_rank(rank)
        ax.plot([x, x], [axis_y - 0.04, y], color=blue, linewidth=1.6)
        ax.plot([x, right_x], [y, y], color=blue, linewidth=1.6)
        ax.text(right_x + 0.10, y, method, color=blue, ha="left", va="center", fontsize=13)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    font_path = r"C:\Windows\Fonts\simsun.ttc"
    chinese_font = FontProperties(fname=font_path) if Path(font_path).exists() else None
    fig.text(0.5, 0.045, "图 2.3  关于 AUC 的 Nemenyi 检验图", ha="center", va="center", fontsize=16, fontproperties=chinese_font)
    fig.text(0.5, 0.015, "Figure 2.3   Nemenyi's test figure on AUC", ha="center", va="center", fontsize=13)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.16)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args):
    data_dir = Path(args.data_dir).resolve()
    workbook = Path(args.auc_table).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_grid(args.k_grid)

    auc_matrix, metadata, _methods = compute_auc_matrix(data_dir, workbook, k_values)
    ranks, pairwise, summary, average_ranks, cd = nemenyi_test(auc_matrix, args.alpha)
    auc_matrix.to_csv(output_dir / "auc_matrix.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(output_dir / "auc_ranks.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output_dir / "nemenyi_pairwise_pvalues.csv", encoding="utf-8-sig")
    summary.to_csv(output_dir / "nemenyi_summary.csv", index=False, encoding="utf-8-sig")
    average_ranks.to_csv(output_dir / "average_ranks.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(output_dir / "gbcof_auc_parameters.csv", index=False, encoding="utf-8-sig")
    draw_cd_diagram(average_ranks, cd, summary, output_dir / "auc_nemenyi_cd.png")
    print(summary.to_string(index=False))
    print(f"Included datasets: {len(auc_matrix)}")
    print(f"Best average rank: {summary.loc[0, 'best_average_rank']}")
    print(f"Saved: {output_dir / 'auc_nemenyi_cd.png'}")


def main():
    parser = argparse.ArgumentParser(description="AUC Friedman/Nemenyi critical-difference diagram")
    parser.add_argument("--data-dir", default=r"F:\实验GBCOF\data")
    parser.add_argument("--auc-table", default=str(WORKBOOK), help="current AUC comparison workbook")
    parser.add_argument("--output-dir", default=r"F:\实验GBCOF\results\nemenyi_auc")
    parser.set_defaults(data_dir=str(DATA_DIR), output_dir=str(OUTPUT_DIR))
    parser.add_argument("--k-grid", default="5:100")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
