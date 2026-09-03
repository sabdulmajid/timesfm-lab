import numpy as np
import pytest

from timesfm_lab.teacher.smoke import array_digest, deterministic_context


def test_deterministic_context_is_reproducible_and_nontrivial() -> None:
    first = deterministic_context(num_variates=3, context_length=64, seed=17)
    second = deterministic_context(num_variates=3, context_length=64, seed=17)

    assert first.shape == (3, 64)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    assert np.all(first > 0)


def test_context_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="num_variates"):
        deterministic_context(num_variates=0, context_length=64, seed=1)
    with pytest.raises(ValueError, match="context_length"):
        deterministic_context(num_variates=1, context_length=31, seed=1)


def test_array_digest_includes_dtype_and_shape() -> None:
    flat = np.arange(6, dtype=np.float32)
    reshaped = flat.reshape(2, 3)
    widened = flat.astype(np.float64)

    assert array_digest(flat) != array_digest(reshaped)
    assert array_digest(flat) != array_digest(widened)
