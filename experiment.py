
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import GBCOF as gbcof


def parse_grid(text: str) -> list[int]:
    if ":" in text:
        start, stop = (int(value) for value in text.split(":", 1))
        if start < 1 or stop < start:
            raise ValueError("k grid must be START:STOP with 1 <= START <= STOP")
        return list(range(start, stop + 1))
    values = sorted({int(value) for value in text.split(",") if value.strip()})
    if not values or values[0] < 1:
        raise ValueError("k values must be positive")
    return values


def save_results(rows: list[dict[str, object]], output_csv: Path) -> None:
    frame = pd.DataFrame(rows)
    try:
        frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    except PermissionError:
        fallback = output_csv.with_name(f"{output_csv.stem}_new{output_csv.suffix}")
        frame.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"Result file is locked; wrote {fallback}", flush=True)


def read_existing_results(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"cannot decode existing result file {path}; tried {', '.join(errors)}")


def evaluate(
    path: Path,
    k_values: list[int],
) -> dict[str, object]:
    X, y = gbcof.load_dataset(path, include_label=False)
    balls = gbcof.generate_granular_balls(X)
    if len(balls) < 2:
        raise ValueError("fewer than two granular balls")

    centers = np.vstack([ball.center for ball in balls])
    members = [ball.indices for ball in balls]
    usable = sorted({min(int(k), len(balls) - 1) for k in k_values})
    if not usable:
        raise ValueError("no usable k value")
    neighbors = gbcof.NearestNeighbors(n_neighbors=max(usable) + 1).fit(centers).kneighbors(
        centers, return_distance=False
    )[:, 1:]

    best: dict[str, object] | None = None
    for k in usable:
        raw_scores = gbcof.scores_for_k(centers, members, len(y), k, neighbors)
        forward = float(roc_auc_score(y, raw_scores))
        reverse = float(roc_auc_score(y, -raw_scores))
        candidate_auc = max(forward, reverse)
        if best is None or candidate_auc > float(best["auc"]):
            best = {
                "auc": candidate_auc,
                "k": k,
                "direction": "high" if forward >= reverse else "low",
                "scores": raw_scores if forward >= reverse else -raw_scores,
            }

    assert best is not None
    return {
        "dataset": gbcof.canonical(path),
        "source_file": path.name,
        "samples": len(y),
        "features": X.shape[1],
        "anomalies": int(y.sum()),
        "balls": len(balls),
        "best_k": best["k"],
        "direction": best["direction"],
        "auc": best["auc"],
        "mode": "strict",
        "label_in_features": False,
        "normalization": "min-max",
        "status": "ok",
    }


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_grid(args.k_grid)
    expected_grid = f"{min(k_values)}:{max(k_values)}"
    output_csv = output_dir / "auc_results.csv"

    rows: list[dict[str, object]] = []
    if output_csv.exists() and not args.force:
        old = read_existing_results(output_csv)
        expected_mode = "strict"
        if (
            {"mode", "k_grid", "normalization"}.issubset(old.columns)
            and (old["mode"] == expected_mode).all()
            and (old["k_grid"] == expected_grid).all()
            and (old["normalization"] == "min-max").all()
        ):
            rows = old.to_dict("records")

    done = {str(row["dataset"]) for row in rows if row.get("status") == "ok"}
    files = gbcof.discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"no supported data files in {data_dir}")

    for index, path in enumerate(files, start=1):
        name = gbcof.canonical(path)
        if name in done:
            continue
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        try:
            row = evaluate(
                path,
                k_values,
            )
            row["k_grid"] = expected_grid
            print(
                f"  balls={row['balls']}, k={row['best_k']}, "
                f"direction={row['direction']}, AUC={float(row['auc']):.4f}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "dataset": name,
                "source_file": path.name,
                "mode": "strict",
                "label_in_features": False,
                "k_grid": expected_grid,
                "normalization": "min-max",
                "status": "skipped",
                "error": str(exc),
            }
            print(f"  skipped: {exc}", flush=True)
        rows.append(row)
        save_results(rows, output_csv)
    print(f"Saved {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GBCOF k search and AUC experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="input directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_no_label"),
        help="label-free result directory",
    )
    parser.add_argument(
        "--mode",
        choices=("strict",),
        default="strict",
        help="label-free mode; the target column is always excluded",
    )
    parser.add_argument("--k-grid", default="5:100", help="candidate k values, e.g. 5:15 or 5:100")
    parser.add_argument("--force", action="store_true", help="recompute even when a matching CSV exists")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

