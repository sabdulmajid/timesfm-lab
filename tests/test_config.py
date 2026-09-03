from pathlib import Path

import pytest

from timesfm_lab.config import ConfigError, load_config


def test_load_config_requires_provenance(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 7\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="dataset_revision, model_revision"):
        load_config(path)


def test_load_config_accepts_explicit_revisions(tmp_path: Path) -> None:
    path = tmp_path / "good.yaml"
    path.write_text(
        "seed: 7\nmodel_revision: abc123\ndataset_revision: def456\n",
        encoding="utf-8",
    )

    assert load_config(path)["seed"] == 7
