import numpy as np
import pytest

from timesfm_lab.distill.data import sample_windows, split_cache_indices


def test_production_sampler_is_deterministic_unique_and_spaced() -> None:
    kwargs = {
        "context_length": 10,
        "horizon": 5,
        "total_windows": 12,
        "seed": 42,
        "mode": "without_replacement_temporally_spaced",
    }
    windows, report = sample_windows([50, 30], **kwargs)
    repeated, _ = sample_windows([50, 30], **kwargs)

    assert windows == repeated
    assert len(windows) == len(set(windows)) == 12
    assert {row for row, _ in windows} == {0, 1}
    assert report["duplicate_windows"] == 0
    assert report["shortfall"] == 0
    for row in (0, 1):
        ends = sorted(end for sampled_row, end in windows if sampled_row == row)
        assert ends[-1] - ends[0] >= 10


def test_production_sampler_exhausts_population_without_duplicates() -> None:
    windows, report = sample_windows(
        [18],
        context_length=10,
        horizon=5,
        total_windows=10,
        seed=7,
        mode="without_replacement_temporally_spaced",
    )

    assert sorted(windows) == [(0, 10), (0, 11), (0, 12), (0, 13)]
    assert report["available_unique_windows"] == 4
    assert report["selected_windows"] == 4
    assert report["shortfall"] == 6


def test_historical_sampler_remains_available() -> None:
    windows, report = sample_windows(
        [16],
        context_length=10,
        horizon=5,
        total_windows=20,
        seed=7,
        mode="with_replacement",
    )

    assert len(windows) == 20
    assert report["duplicate_windows"] > 0


def test_series_split_holds_out_whole_rows() -> None:
    rows = np.repeat(np.arange(4), 10)
    ends = np.tile(np.arange(20, 30), 4)
    split, report = split_cache_indices(
        rows,
        ends,
        context_length=5,
        horizon=2,
        validation_fraction=0.25,
        seed=42,
        mode="series_or_time",
    )

    training_rows = set(rows[split["training"]])
    validation_rows = set(rows[split["validation"]])
    assert training_rows.isdisjoint(validation_rows)
    assert report["mode"] == "held_out_series"


def test_blocked_split_has_no_context_or_target_overlap() -> None:
    rows = np.zeros(40, dtype=np.int64)
    ends = np.arange(10, 50)
    context = 10
    horizon = 5
    split, report = split_cache_indices(
        rows,
        ends,
        context_length=context,
        horizon=horizon,
        validation_fraction=0.2,
        seed=42,
        mode="series_or_time",
    )

    assert report["mode"] == "blocked_time"
    for train_index in split["training"]:
        train_interval = (ends[train_index] - context, ends[train_index] + horizon)
        for validation_index in split["validation"]:
            validation_interval = (
                ends[validation_index] - context,
                ends[validation_index] + horizon,
            )
            assert train_interval[1] <= validation_interval[0]


def test_blocked_split_rejects_too_short_cache() -> None:
    with pytest.raises(ValueError, match="left no training windows"):
        split_cache_indices(
            np.zeros(3),
            np.array([10, 11, 12]),
            context_length=10,
            horizon=2,
            validation_fraction=0.33,
            seed=42,
            mode="blocked_time",
        )
