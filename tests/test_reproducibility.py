import numpy as np
import pytest

from enzymeopt.randomness import SeedManager, derive_seed, make_rng


def test_same_master_seed_and_labels_reproduce_sequence() -> None:
    first = make_rng(42, "replicate", 3, "random", "noise").normal(size=20)
    second = make_rng(42, "replicate", 3, "random", "noise").normal(size=20)

    np.testing.assert_array_equal(first, second)


def test_distinct_labels_produce_distinct_streams() -> None:
    random_values = make_rng(42, "replicate", 3, "random").integers(0, 2**32, size=20)
    optimal_values = make_rng(42, "replicate", 3, "d_optimal").integers(0, 2**32, size=20)

    assert not np.array_equal(random_values, optimal_values)


def test_strategy_iteration_order_does_not_change_streams() -> None:
    manager = SeedManager(20260915)

    forward = {
        strategy: manager.rng_for("replicate", 7, strategy, "noise").normal(size=10)
        for strategy in ("random", "log_spaced", "d_optimal")
    }
    reverse = {
        strategy: manager.rng_for("replicate", 7, strategy, "noise").normal(size=10)
        for strategy in reversed(("random", "log_spaced", "d_optimal"))
    }

    for strategy in forward:
        np.testing.assert_array_equal(forward[strategy], reverse[strategy])


def test_label_types_cannot_collide() -> None:
    assert derive_seed(1, 7) != derive_seed(1, "7")


@pytest.mark.parametrize("seed", [-1, 2**128, True])
def test_invalid_master_seed_is_rejected(seed: object) -> None:
    with pytest.raises(ValueError, match="master_seed"):
        SeedManager(seed)  # type: ignore[arg-type]


def test_invalid_seed_label_is_rejected() -> None:
    with pytest.raises(TypeError, match="strings or integers"):
        derive_seed(1, 1.5)  # type: ignore[arg-type]
