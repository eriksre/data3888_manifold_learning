from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from recover_stock_price import (
    BOOK_COLS,
    PRICE_COLS,
    extract_stock_id,
    recover_time_id_order,
)


def robust_anchor_and_confidence(
    block: pd.DataFrame,
    tick_size: float = 0.01,
) -> tuple[float, float]:
    # Estimate one noisy absolute price anchor for the bucket by inverting the
    # normalized tick size, and attach a confidence based on how consistent the
    # inferred tick samples are.
    diff = block[PRICE_COLS].diff().abs().to_numpy(dtype=float).ravel()
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if diff.size == 0:
        return np.nan, 0.0

    min_diff = diff.min()
    if not np.isfinite(min_diff) or min_diff <= 0:
        return np.nan, 0.0

    n_ticks = np.round(diff / min_diff)
    n_ticks = np.where(n_ticks <= 0, 1.0, n_ticks)
    tick_samples = diff / n_ticks
    tick_samples = tick_samples[np.isfinite(tick_samples) & (tick_samples > 0)]
    if tick_samples.size == 0:
        return np.nan, 0.0

    tick_med = float(np.median(tick_samples))
    log_dev = np.abs(np.log(tick_samples / tick_med))
    inliers = tick_samples[log_dev <= 0.15]
    if inliers.size >= 5:
        tick_est = float(np.mean(inliers))
        dispersion = float(np.std(np.log(inliers / tick_est + 1e-12)))
        sample_count = inliers.size
    else:
        tick_est = tick_med
        dispersion = float(np.std(np.log(tick_samples / tick_est + 1e-12)))
        sample_count = tick_samples.size

    if not np.isfinite(tick_est) or tick_est <= 0:
        return np.nan, 0.0

    anchor = float(tick_size / tick_est)
    confidence = float(np.sqrt(sample_count) / (1.0 + 30.0 * dispersion))
    confidence = max(0.05, min(confidence, 20.0))
    return anchor, confidence


def summarize_stock_file(
    path: Path,
    tick_size: float = 0.01,
) -> pd.DataFrame:
    stock_id = extract_stock_id(path)
    df = pd.read_csv(path, usecols=BOOK_COLS)
    num = (
        df["bid_price1"] * df["ask_size1"]
        + df["ask_price1"] * df["bid_size1"]
        + df["bid_price2"] * df["ask_size2"]
        + df["ask_price2"] * df["bid_size2"]
    )
    den = df["bid_size1"] + df["ask_size1"] + df["bid_size2"] + df["ask_size2"]
    df["wap"] = num / den

    rows: list[dict[str, float]] = []

    for time_id, block in df.groupby("time_id", sort=True):
        block = block.sort_values("seconds_in_bucket")
        anchor, confidence = robust_anchor_and_confidence(block, tick_size=tick_size)

        # Within a bucket, only relative movement matters for v2. We keep the
        # normalized WAP shape and later solve for one opening level per bucket.
        wap = (
            block.set_index("seconds_in_bucket")["wap"]
            .reindex(np.arange(600))
            .ffill()
            .bfill()
        )
        if not np.isfinite(wap.iloc[0]) or wap.iloc[0] <= 0:
            close_ratio = np.nan
            norm_path = np.full(600, np.nan)
        else:
            norm_path = (wap / wap.iloc[0]).to_numpy(dtype=float)
            close_ratio = float(norm_path[-1])

        rows.append(
            {
                "time_id": time_id,
                "stock_id": stock_id,
                "anchor": anchor,
                "anchor_conf": confidence,
                "close_ratio": close_ratio,
            }
        )
    return pd.DataFrame(rows)


def load_normalized_path(stock_path: Path) -> pd.DataFrame:
    df = pd.read_csv(stock_path)
    num = (
        df["bid_price1"] * df["ask_size1"]
        + df["ask_price1"] * df["bid_size1"]
        + df["bid_price2"] * df["ask_size2"]
        + df["ask_price2"] * df["bid_size2"]
    )
    den = df["bid_size1"] + df["ask_size1"] + df["bid_size2"] + df["ask_size2"]
    df["wap"] = num / den

    path_rows: list[pd.DataFrame] = []
    for time_id, block in df.groupby("time_id", sort=True):
        block = block.sort_values("seconds_in_bucket")
        first_wap = float(block["wap"].iloc[0]) if not block.empty else np.nan
        if not np.isfinite(first_wap) or first_wap <= 0:
            norm_path = np.full(len(block), np.nan)
        else:
            norm_path = (block["wap"] / first_wap).to_numpy(dtype=float)
        out_block = block.copy()
        out_block["norm_wap"] = norm_path
        path_rows.append(out_block)
    return pd.concat(path_rows, ignore_index=True)


def build_panel_data(
    book_dir: Path,
    tick_size: float = 0.01,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, Path]]:
    stock_paths = sorted(book_dir.glob("stock_*.csv"))
    if not stock_paths:
        raise FileNotFoundError(f"No stock_*.csv files found in {book_dir}")

    if workers <= 1:
        summaries = [summarize_stock_file(path, tick_size=tick_size) for path in stock_paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            summaries = list(executor.map(summarize_stock_file, stock_paths, [tick_size] * len(stock_paths)))

    panel = pd.concat(summaries, ignore_index=True)
    # These matrices are indexed by time_id x stock_id and are the shared panel
    # inputs for global ordering plus stock-level latent-price reconstruction.
    anchor_matrix = panel.pivot(index="time_id", columns="stock_id", values="anchor").sort_index()
    anchor_conf_matrix = panel.pivot(index="time_id", columns="stock_id", values="anchor_conf").sort_index()
    close_ratio_matrix = panel.pivot(index="time_id", columns="stock_id", values="close_ratio").sort_index()
    stock_path_map = {extract_stock_id(path): path for path in stock_paths}
    return anchor_matrix, anchor_conf_matrix, close_ratio_matrix, stock_path_map


def compute_panel_market_jump(
    anchor_matrix: pd.DataFrame,
    close_ratio_matrix: pd.DataFrame,
    ordered_time_ids: np.ndarray,
    use_panel_jump: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_edges = len(ordered_time_ids) - 1
    if not use_panel_jump:
        return np.zeros(n_edges, dtype=float), np.ones(n_edges, dtype=float)

    ordered_anchor = anchor_matrix.reindex(ordered_time_ids).to_numpy(dtype=float)
    ordered_ratio = close_ratio_matrix.reindex(ordered_time_ids).to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        implied_jump = (
            np.log(ordered_anchor[1:])
            - np.log(ordered_anchor[:-1])
            - np.log(ordered_ratio[:-1])
        )

    gap = np.nanmedian(implied_jump, axis=1)
    gap = np.where(np.isfinite(gap), gap, 0.0)

    mad = np.nanmedian(np.abs(implied_jump - gap[:, None]), axis=1)
    mad = np.where(np.isfinite(mad), mad, np.nanmedian(mad[np.isfinite(mad)]))
    mad = np.where(np.isfinite(mad), mad, 0.02)
    finite_count = np.sum(np.isfinite(implied_jump), axis=1).astype(float)
    mad_scale = float(np.nanmedian(mad[np.isfinite(mad)]))
    if not np.isfinite(mad_scale) or mad_scale <= 0:
        mad_scale = 0.02

    confidence = finite_count / (1.0 + (mad / mad_scale) ** 2)
    finite_conf = confidence[np.isfinite(confidence) & (confidence > 0)]
    norm = float(np.nanmedian(finite_conf)) if finite_conf.size else 1.0
    if not np.isfinite(norm) or norm <= 0:
        norm = 1.0
    confidence = np.clip(confidence / norm, 0.05, 20.0)
    return gap, confidence


def solve_log_open_levels(
    log_anchor: np.ndarray,
    anchor_weight: np.ndarray,
    log_close_ratio: np.ndarray,
    market_gap: np.ndarray,
    continuity_weight: np.ndarray,
    lambda_cont: float,
) -> np.ndarray:
    # Solve a 1D smoothing problem over bucket opening levels in log-price
    # space. The solution balances local tick-derived anchors against continuity
    # from one bucket close to the next bucket open.
    n = len(log_anchor)
    diag_main = anchor_weight.astype(float).copy()
    rhs = anchor_weight * log_anchor
    if n == 1:
        return rhs / np.maximum(diag_main, 1e-8)

    edge_w = lambda_cont * continuity_weight
    off = -edge_w
    b = log_close_ratio[:-1] + market_gap

    diag_main[:-1] += edge_w
    diag_main[1:] += edge_w

    rhs[0] += -edge_w[0] * b[0]
    rhs[-1] += edge_w[-1] * b[-1]
    if n > 2:
        rhs[1:-1] += edge_w[:-1] * b[:-1] - edge_w[1:] * b[1:]

    system = diags(
        diagonals=[off, diag_main, off],
        offsets=[-1, 0, 1],
        format="csc",
    )
    return spsolve(system, rhs)


def reconstruct_from_levels(
    ordered_time_ids: np.ndarray,
    target_norm: pd.DataFrame,
    open_levels: np.ndarray,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    original_columns = [col for col in target_norm.columns if col not in {"wap", "norm_wap"}]
    block_lookup = {
        time_id: grp.sort_values("seconds_in_bucket").copy()
        for time_id, grp in target_norm.groupby("time_id", sort=False)
    }
    rows: list[pd.DataFrame] = []
    for rank, (time_id, level) in enumerate(zip(ordered_time_ids, open_levels)):
        # Re-expand the normalized intra-bucket path using the solved bucket
        # opening level to recover the original per-row price path.
        block = block_lookup[int(time_id)].copy()
        block["recovered_rank"] = rank
        block["recovered_price"] = level * block["norm_wap"].to_numpy(dtype=float)
        block["base_price"] = level
        rows.append(block)
    result = pd.concat(rows, ignore_index=True)
    result["global_second"] = np.arange(len(result))
    result = result[original_columns + ["global_second", "recovered_rank", "recovered_price", "base_price"]]
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


def stock_jump_stats_from_path(recovered_path: pd.DataFrame) -> dict[str, float]:
    bucket = recovered_path.groupby("time_id").agg(
        open=("recovered_price", "first"),
        close=("recovered_price", "last"),
    )
    order = (
        recovered_path[["recovered_rank", "time_id"]]
        .drop_duplicates()
        .sort_values("recovered_rank")["time_id"]
        .to_numpy()
    )
    ordered = bucket.reindex(order)
    gaps = np.abs(ordered["close"].to_numpy(dtype=float)[:-1] - ordered["open"].to_numpy(dtype=float)[1:])
    return {
        "mean": float(np.mean(gaps)),
        "median": float(np.median(gaps)),
        "p95": float(np.quantile(gaps, 0.95)),
        "p99": float(np.quantile(gaps, 0.99)),
        "max": float(np.max(gaps)),
        "gt5": int(np.sum(gaps > 5)),
        "gt10": int(np.sum(gaps > 10)),
    }


def candidate_score(stats: dict[str, float]) -> float:
    return (
        stats["mean"]
        + 0.35 * stats["p95"]
        + 0.45 * stats["p99"]
        + 0.03 * stats["gt5"]
        + 0.6 * stats["gt10"]
    )


def run_candidate(
    anchor_matrix: pd.DataFrame,
    anchor_conf_matrix: pd.DataFrame,
    close_ratio_matrix: pd.DataFrame,
    stock_path: Path,
    target_stock_id: int,
    n_neighbors: int,
    lambda_cont: float,
    use_panel_jump: bool,
    use_anchor_conf: bool,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    target_norm = load_normalized_path(stock_path)
    # The whole panel determines one global permutation of unique time_ids.
    time_order = recover_time_id_order(anchor_matrix, n_neighbors=n_neighbors)
    ordered_time_ids = time_order["time_id"].to_numpy()

    target_anchor = anchor_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)
    target_conf = anchor_conf_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)
    target_ratio = close_ratio_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)

    finite_anchor = np.isfinite(target_anchor) & (target_anchor > 0)
    anchor_fill = float(np.nanmedian(target_anchor[finite_anchor]))
    target_anchor = np.where(finite_anchor, target_anchor, anchor_fill)

    if use_anchor_conf:
        conf = np.where(np.isfinite(target_conf) & (target_conf > 0), target_conf, np.nanmedian(target_conf[np.isfinite(target_conf) & (target_conf > 0)]))
        conf = np.clip(conf, 0.05, 20.0)
        conf /= np.nanmedian(conf)
    else:
        conf = np.ones_like(target_anchor, dtype=float)

    ratio_finite = np.isfinite(target_ratio) & (target_ratio > 0)
    target_ratio = np.where(ratio_finite, target_ratio, 1.0)

    market_gap, continuity_conf = compute_panel_market_jump(
        anchor_matrix=anchor_matrix,
        close_ratio_matrix=close_ratio_matrix,
        ordered_time_ids=ordered_time_ids,
        use_panel_jump=use_panel_jump,
    )

    log_anchor = np.log(target_anchor)
    log_close_ratio = np.log(target_ratio)
    log_open_levels = solve_log_open_levels(
        log_anchor=log_anchor,
        anchor_weight=conf,
        log_close_ratio=log_close_ratio,
        market_gap=market_gap,
        continuity_weight=continuity_conf,
        lambda_cont=lambda_cont,
    )
    open_levels = np.exp(log_open_levels)

    recovered_path = reconstruct_from_levels(
        ordered_time_ids=ordered_time_ids,
        target_norm=target_norm,
        open_levels=open_levels,
    )
    stats = stock_jump_stats_from_path(recovered_path)
    stats["score"] = candidate_score(stats)
    stats["n_neighbors"] = n_neighbors
    stats["lambda_cont"] = lambda_cont
    stats["use_panel_jump"] = int(use_panel_jump)
    stats["use_anchor_conf"] = int(use_anchor_conf)
    return stats, time_order, recovered_path


def reconstruct_single_stock_with_params(
    anchor_matrix: pd.DataFrame,
    anchor_conf_matrix: pd.DataFrame,
    close_ratio_matrix: pd.DataFrame,
    stock_path: Path,
    target_stock_id: int,
    n_neighbors: int,
    lambda_cont: float,
    use_panel_jump: bool,
    use_anchor_conf: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    target_norm = load_normalized_path(stock_path)
    time_order = recover_time_id_order(anchor_matrix, n_neighbors=n_neighbors)
    ordered_time_ids = time_order["time_id"].to_numpy()

    target_anchor = anchor_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)
    target_conf = anchor_conf_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)
    target_ratio = close_ratio_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)

    finite_anchor = np.isfinite(target_anchor) & (target_anchor > 0)
    anchor_fill = float(np.nanmedian(target_anchor[finite_anchor]))
    target_anchor = np.where(finite_anchor, target_anchor, anchor_fill)

    if use_anchor_conf:
        finite_conf = np.isfinite(target_conf) & (target_conf > 0)
        conf_fill = float(np.nanmedian(target_conf[finite_conf])) if finite_conf.any() else 1.0
        conf = np.where(finite_conf, target_conf, conf_fill)
        conf = np.clip(conf, 0.05, 20.0)
        conf /= np.nanmedian(conf)
    else:
        conf = np.ones_like(target_anchor, dtype=float)

    ratio_finite = np.isfinite(target_ratio) & (target_ratio > 0)
    target_ratio = np.where(ratio_finite, target_ratio, 1.0)

    market_gap, continuity_conf = compute_panel_market_jump(
        anchor_matrix=anchor_matrix,
        close_ratio_matrix=close_ratio_matrix,
        ordered_time_ids=ordered_time_ids,
        use_panel_jump=use_panel_jump,
    )

    log_anchor = np.log(target_anchor)
    log_close_ratio = np.log(target_ratio)
    log_open_levels = solve_log_open_levels(
        log_anchor=log_anchor,
        anchor_weight=conf,
        log_close_ratio=log_close_ratio,
        market_gap=market_gap,
        continuity_weight=continuity_conf,
        lambda_cont=lambda_cont,
    )
    open_levels = np.exp(log_open_levels)
    recovered_path = reconstruct_from_levels(
        ordered_time_ids=ordered_time_ids,
        target_norm=target_norm,
        open_levels=open_levels,
    )
    stats = stock_jump_stats_from_path(recovered_path)
    stats["target_stock_id"] = target_stock_id
    return time_order, recovered_path, stats


def reconstruct_single_stock_with_order(
    anchor_matrix: pd.DataFrame,
    anchor_conf_matrix: pd.DataFrame,
    close_ratio_matrix: pd.DataFrame,
    stock_path: Path,
    target_stock_id: int,
    ordered_time_ids: np.ndarray,
    lambda_cont: float,
    use_panel_jump: bool,
    use_anchor_conf: bool,
) -> pd.DataFrame:
    target_norm = load_normalized_path(stock_path)

    target_anchor = anchor_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)
    target_conf = anchor_conf_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)
    target_ratio = close_ratio_matrix[target_stock_id].reindex(ordered_time_ids).to_numpy(dtype=float)

    finite_anchor = np.isfinite(target_anchor) & (target_anchor > 0)
    anchor_fill = float(np.nanmedian(target_anchor[finite_anchor]))
    target_anchor = np.where(finite_anchor, target_anchor, anchor_fill)

    if use_anchor_conf:
        finite_conf = np.isfinite(target_conf) & (target_conf > 0)
        conf_fill = float(np.nanmedian(target_conf[finite_conf])) if finite_conf.any() else 1.0
        conf = np.where(finite_conf, target_conf, conf_fill)
        conf = np.clip(conf, 0.05, 20.0)
        conf /= np.nanmedian(conf)
    else:
        conf = np.ones_like(target_anchor, dtype=float)

    ratio_finite = np.isfinite(target_ratio) & (target_ratio > 0)
    target_ratio = np.where(ratio_finite, target_ratio, 1.0)

    market_gap, continuity_conf = compute_panel_market_jump(
        anchor_matrix=anchor_matrix,
        close_ratio_matrix=close_ratio_matrix,
        ordered_time_ids=ordered_time_ids,
        use_panel_jump=use_panel_jump,
    )

    log_anchor = np.log(target_anchor)
    log_close_ratio = np.log(target_ratio)
    log_open_levels = solve_log_open_levels(
        log_anchor=log_anchor,
        anchor_weight=conf,
        log_close_ratio=log_close_ratio,
        market_gap=market_gap,
        continuity_weight=continuity_conf,
        lambda_cont=lambda_cont,
    )
    open_levels = np.exp(log_open_levels)
    return reconstruct_from_levels(
        ordered_time_ids=ordered_time_ids,
        target_norm=target_norm,
        open_levels=open_levels,
    )


def infer_default_workers() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, min(8, cpu_count - 1))


def run_v2(
    book_dir: Path,
    output_csv: Path,
    order_csv: Path,
    target_stock_id: int = 1,
    workers: int | None = None,
    tick_size: float = 0.01,
    n_neighbors_grid: list[int] | tuple[int, ...] = (40, 60, 80),
    lambda_cont_grid: list[float] | tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 20.0, 40.0),
    use_panel_jump_grid: list[bool] | tuple[bool, ...] = (False, True),
    use_anchor_conf_grid: list[bool] | tuple[bool, ...] = (False, True),
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    if workers is None:
        workers = infer_default_workers()

    anchor_matrix, anchor_conf_matrix, close_ratio_matrix, stock_path_map = build_panel_data(
        book_dir,
        tick_size=tick_size,
        workers=workers,
    )
    stock_path = stock_path_map[target_stock_id]

    # Single-stock mode searches over reconstruction hyperparameters and keeps
    # the candidate with the cleanest bucket boundary behavior.
    candidates: list[tuple[dict[str, float], pd.DataFrame, pd.DataFrame]] = []
    for n_neighbors in n_neighbors_grid:
        for lambda_cont in lambda_cont_grid:
            for use_panel_jump in use_panel_jump_grid:
                for use_anchor_conf in use_anchor_conf_grid:
                    stats, time_order, recovered_path = run_candidate(
                        anchor_matrix=anchor_matrix,
                        anchor_conf_matrix=anchor_conf_matrix,
                        close_ratio_matrix=close_ratio_matrix,
                        stock_path=stock_path,
                        target_stock_id=target_stock_id,
                        n_neighbors=n_neighbors,
                        lambda_cont=lambda_cont,
                        use_panel_jump=use_panel_jump,
                        use_anchor_conf=use_anchor_conf,
                    )
                    candidates.append((stats, time_order, recovered_path))
                    print(
                        "candidate",
                        {
                            "n_neighbors": n_neighbors,
                            "lambda_cont": lambda_cont,
                            "panel_jump": use_panel_jump,
                            "anchor_conf": use_anchor_conf,
                            "mean": round(stats["mean"], 4),
                            "p95": round(stats["p95"], 4),
                            "p99": round(stats["p99"], 4),
                            "gt5": stats["gt5"],
                            "gt10": stats["gt10"],
                            "score": round(stats["score"], 4),
                        },
                    )

    best_stats, best_order, best_path = min(candidates, key=lambda x: x[0]["score"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    order_csv.parent.mkdir(parents=True, exist_ok=True)
    best_path.to_csv(output_csv, index=False)
    best_order.to_csv(order_csv, index=False)

    print("\nbest_candidate", best_stats)
    print(f"Wrote best recovered path to {output_csv}")
    print(f"Wrote best recovered order to {order_csv}")

    return best_stats, best_order, best_path


def run_v2_single_file(
    input_csv: Path,
    output_csv: Path,
    order_csv: Path,
    workers: int | None = None,
    tick_size: float = 0.01,
    n_neighbors_grid: list[int] | tuple[int, ...] = (40, 60, 80),
    lambda_cont_grid: list[float] | tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 20.0, 40.0),
    use_panel_jump_grid: list[bool] | tuple[bool, ...] = (False, True),
    use_anchor_conf_grid: list[bool] | tuple[bool, ...] = (False, True),
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    input_csv = Path(input_csv)
    return run_v2(
        book_dir=input_csv.parent,
        output_csv=output_csv,
        order_csv=order_csv,
        target_stock_id=extract_stock_id(input_csv),
        workers=workers,
        tick_size=tick_size,
        n_neighbors_grid=n_neighbors_grid,
        lambda_cont_grid=lambda_cont_grid,
        use_panel_jump_grid=use_panel_jump_grid,
        use_anchor_conf_grid=use_anchor_conf_grid,
    )


def run_v2_folder(
    book_dir: Path,
    output_dir: Path,
    order_csv: Path,
    n_neighbors: int,
    lambda_cont: float,
    use_panel_jump: bool,
    use_anchor_conf: bool,
    workers: int | None = None,
    tick_size: float = 0.01,
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    if workers is None:
        workers = infer_default_workers()

    anchor_matrix, anchor_conf_matrix, close_ratio_matrix, stock_path_map = build_panel_data(
        book_dir,
        tick_size=tick_size,
        workers=workers,
    )
    time_order = recover_time_id_order(anchor_matrix, n_neighbors=n_neighbors)
    ordered_time_ids = time_order["time_id"].to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    order_csv.parent.mkdir(parents=True, exist_ok=True)
    time_order.to_csv(order_csv, index=False)

    stats_by_stock: dict[int, dict[str, float]] = {}
    for stock_id, stock_path in sorted(stock_path_map.items()):
        # Folder mode reuses one shared global order, then reconstructs each
        # stock independently against that same ordering.
        recovered_path = reconstruct_single_stock_with_order(
            anchor_matrix=anchor_matrix,
            anchor_conf_matrix=anchor_conf_matrix,
            close_ratio_matrix=close_ratio_matrix,
            stock_path=stock_path,
            target_stock_id=stock_id,
            ordered_time_ids=ordered_time_ids,
            lambda_cont=lambda_cont,
            use_panel_jump=use_panel_jump,
            use_anchor_conf=use_anchor_conf,
        )
        stats = stock_jump_stats_from_path(recovered_path)
        stats["target_stock_id"] = stock_id
        out_csv = output_dir / f"recovered_stock_{stock_id}_prices_v2.csv"
        recovered_path.to_csv(out_csv, index=False)
        stats_by_stock[stock_id] = stats
        print(
            "wrote",
            {
                "stock_id": stock_id,
                "output": str(out_csv),
                "mean": round(stats["mean"], 4),
                "p95": round(stats["p95"], 4),
                "p99": round(stats["p99"], 4),
            },
        )

    return time_order, stats_by_stock


if __name__ == "__main__":

    MODE = "single" # Change to "folder" if you want to use folder mode.
    
    # Single-stock mode
    INPUT_CSV = Path("individual_book_train/stock_2.csv")
    OUTPUT_CSV = Path("recovered_stock_2_prices_v2.csv")
    ORDER_CSV = Path("recovered_time_id_order_v2.csv")

    # # Folder mode. Uncomment the lines below, and comment out the single-stock mode lines above. 
    # BOOK_DIR = Path("individual_book_train")
    # OUTPUT_DIR = Path("recovered_v2_all")

    # Single-stock mode parameters
    WORKERS = 6
    TICK_SIZE = 0.01
    N_NEIGHBORS_GRID = [40, 60, 80]
    LAMBDA_CONT_GRID = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0]
    USE_PANEL_JUMP_GRID = [False, True]
    USE_ANCHOR_CONF_GRID = [False, True]

    # Folder mode uses one fixed parameter set for every stock.
    FOLDER_N_NEIGHBORS = 60
    FOLDER_LAMBDA_CONT = 40.0
    FOLDER_USE_PANEL_JUMP = False
    FOLDER_USE_ANCHOR_CONF = True

    if MODE == "single":
        run_v2_single_file(
            input_csv=INPUT_CSV,
            output_csv=OUTPUT_CSV,
            order_csv=ORDER_CSV,
            workers=WORKERS,
            tick_size=TICK_SIZE,
            n_neighbors_grid=N_NEIGHBORS_GRID,
            lambda_cont_grid=LAMBDA_CONT_GRID,
            use_panel_jump_grid=USE_PANEL_JUMP_GRID,
            use_anchor_conf_grid=USE_ANCHOR_CONF_GRID,
        )
    elif MODE == "folder":
        run_v2_folder(
            book_dir=BOOK_DIR,
            output_dir=OUTPUT_DIR,
            order_csv=ORDER_CSV,
            n_neighbors=FOLDER_N_NEIGHBORS,
            lambda_cont=FOLDER_LAMBDA_CONT,
            use_panel_jump=FOLDER_USE_PANEL_JUMP,
            use_anchor_conf=FOLDER_USE_ANCHOR_CONF,
            workers=WORKERS,
            tick_size=TICK_SIZE,
        )
    else:
        raise ValueError("MODE must be either 'single' or 'folder'")
