"""Weights & Biases integration — optional, activates when WANDB_API_KEY is set.

All public functions are no-ops when wandb is not configured, so the rest of
the codebase can call them unconditionally without try/except guards.

Usage in train.py:
    from src.utils.wandb_logger import wandb_init, wandb_log, wandb_finish
    run = wandb_init(config={...})
    wandb_log({"metric": value})
    wandb_finish()
"""
from __future__ import annotations

import os
from typing import Any


def _is_enabled() -> bool:
    return bool(os.getenv("WANDB_API_KEY", "").strip())


def wandb_init(
    config: dict[str, Any] | None = None,
    job_type: str = "train",
    tags: list[str] | None = None,
) -> Any:
    """Initialize a wandb run. Returns the run object (or None if disabled).

    Args:
        config: Hyperparameters and metadata to log.
        job_type: wandb job type label (e.g. "train", "backtest").
        tags: List of string tags for the run.

    Returns:
        wandb.Run object if enabled, else None.
    """
    if not _is_enabled():
        return None
    try:
        import wandb  # type: ignore[import]
        project = os.getenv("WANDB_PROJECT", "worldcup-predictor")
        run = wandb.init(
            project=project,
            config=config or {},
            job_type=job_type,
            tags=tags or [],
            reinit=True,
        )
        return run
    except Exception as e:
        print(f"[wandb] init failed (continuing without tracking): {e}")
        return None


def wandb_log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log a dict of metrics to the current wandb run.

    Args:
        metrics: Dict of metric name → value (scalar, image, table, etc.)
        step: Optional global step counter.
    """
    if not _is_enabled():
        return
    try:
        import wandb  # type: ignore[import]
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def wandb_log_table(
    name: str,
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    """Log a table to wandb (e.g. feature importances, backtest details).

    Args:
        name: Table name shown in the wandb UI.
        columns: Column headers.
        rows: List of rows, each a list matching the columns.
    """
    if not _is_enabled():
        return
    try:
        import wandb  # type: ignore[import]
        if wandb.run is not None:
            table = wandb.Table(columns=columns, data=rows)
            wandb.log({name: table})
    except Exception:
        pass


def wandb_finish() -> None:
    """Finish the current wandb run."""
    if not _is_enabled():
        return
    try:
        import wandb  # type: ignore[import]
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
