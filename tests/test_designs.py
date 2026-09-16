from math import exp, log

import numpy as np
import pytest

from enzymeopt.designs import (
    ConcentrationStrategy,
    DesignState,
    LogSpacedDesign,
    RandomDesign,
    generate_design,
)
from enzymeopt.randomness import make_rng
from enzymeopt.designs import DOptimalDesign
from enzymeopt.information import fisher_information


def test_random_design_samples_uniformly_in_log_space() -> None:
    seed_labels = (44, "random-design")
    actual = generate_design(
        RandomDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=8,
        rng=make_rng(*seed_labels),
    )
    reference_rng = make_rng(*seed_labels)
    expected = np.exp(reference_rng.uniform(log(0.05), log(20.0), size=8))

    np.testing.assert_array_equal(actual, expected)


def test_random_design_is_reproducible_and_within_bounds() -> None:
    first = generate_design(
        RandomDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=24,
        rng=make_rng(10, "random"),
    )
    second = generate_design(
        RandomDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=24,
        rng=make_rng(10, "random"),
    )

    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0.05)
    assert np.all(first <= 20.0)


def test_random_design_requires_explicit_rng() -> None:
    with pytest.raises(TypeError, match="explicit numpy.random.Generator"):
        generate_design(
            RandomDesign(),
            substrate_min=0.05,
            substrate_max=20.0,
            measurement_budget=3,
        )


def test_log_spaced_design_includes_endpoints_and_equal_log_intervals() -> None:
    design = generate_design(
        LogSpacedDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=8,
    )

    assert design[0] == pytest.approx(0.05)
    assert design[-1] == pytest.approx(20.0)
    np.testing.assert_allclose(np.diff(np.log(design)), np.diff(np.log(design))[0])


def test_log_spaced_design_is_independent_for_each_budget() -> None:
    four_points = generate_design(
        LogSpacedDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=4,
    )
    six_points = generate_design(
        LogSpacedDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=6,
    )

    np.testing.assert_allclose(four_points, np.geomspace(0.05, 20.0, 4))
    np.testing.assert_allclose(six_points, np.geomspace(0.05, 20.0, 6))
    assert not np.array_equal(four_points, six_points[:4])


def test_log_spaced_design_does_not_consume_rng() -> None:
    used_rng = make_rng(18, "log-spaced")
    untouched_rng = make_rng(18, "log-spaced")

    generate_design(
        LogSpacedDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=3,
        rng=used_rng,
    )

    np.testing.assert_array_equal(used_rng.normal(size=5), untouched_rng.normal(size=5))


def test_both_strategies_share_interface_and_allow_repeated_concentrations() -> None:
    for strategy in (RandomDesign(), LogSpacedDesign()):
        assert isinstance(strategy, ConcentrationStrategy)

    state = DesignState(
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=3,
        selected_concentrations=(1.0, 1.0),
    )

    assert state.selected_concentrations == (1.0, 1.0)
    assert not state.is_complete


def test_generate_design_can_continue_existing_measurements() -> None:
    design = generate_design(
        RandomDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=4,
        selected_concentrations=[0.05, 1.0],
        rng=make_rng(20, "continue"),
    )

    np.testing.assert_array_equal(design[:2], [0.05, 1.0])
    assert design.size == 4


def test_log_spaced_design_can_continue_its_existing_prefix() -> None:
    full_design = np.geomspace(0.05, 20.0, 4)

    design = generate_design(
        LogSpacedDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=4,
        selected_concentrations=full_design[:2],
    )

    np.testing.assert_allclose(design, full_design)


def test_log_spaced_design_rejects_non_prefix_history() -> None:
    with pytest.raises(ValueError, match="prefix"):
        generate_design(
            LogSpacedDesign(),
            substrate_min=0.05,
            substrate_max=20.0,
            measurement_budget=4,
            selected_concentrations=[0.05, 1.0],
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"substrate_min": 0.0},
        {"substrate_min": 2.0, "substrate_max": 1.0},
        {"measurement_budget": 1},
        {"selected_concentrations": (0.01,)},
        {"candidate_concentrations": (21.0,)},
    ],
)
def test_design_state_rejects_invalid_ranges(arguments: dict[str, object]) -> None:
    defaults: dict[str, object] = {
        "substrate_min": 0.05,
        "substrate_max": 20.0,
        "measurement_budget": 3,
    }
    defaults.update(arguments)

    with pytest.raises(ValueError):
        DesignState(**defaults)  # type: ignore[arg-type]


def test_generated_design_is_read_only() -> None:
    design = generate_design(
        LogSpacedDesign(),
        substrate_min=0.05,
        substrate_max=20.0,
        measurement_budget=3,
    )

    with pytest.raises(ValueError, match="read-only"):
        design[0] = exp(log(1.0))


def test_d_optimal_matches_independent_candidate_enumeration():
    state = DesignState(0.05, 20, 8, (0.05, 1., 20.),
                        tuple(np.geomspace(0.05, 20, 200)))
    strategy = DOptimalDesign(km=1.2, vmax=2, noise_std=0.05)
    expected = []
    for candidate in state.candidate_concentrations:
        s = np.array((*state.selected_concentrations, candidate))
        jac = np.column_stack((-2*s/(1.2+s)**2, s/(1.2+s))) / 0.05
        expected.append(np.linalg.slogdet(jac.T @ jac)[1])
    np.testing.assert_allclose(strategy.score_candidates(state), expected)
    assert strategy.select_next(state) == state.candidate_concentrations[np.argmax(expected)]
    np.testing.assert_allclose(strategy.current_information(state),
        fisher_information(state.selected_concentrations, km=1.2, vmax=2, noise_std=0.05))


def test_d_optimal_singular_history_can_be_resolved():
    state = DesignState(0.1, 10, 4, (1., 1.), (1., 10.))
    assert DOptimalDesign(1, 1).select_next(state) == 10


def test_d_optimal_rejects_uninformative_and_missing_candidates():
    for state in [DesignState(0.1, 10, 4, (), (1., 2.)),
                  DesignState(0.1, 10, 4, (1., 1.), (1.,)),
                  DesignState(0.1, 10, 4)]:
        with pytest.raises(ValueError):
            DOptimalDesign(1, 1).select_next(state)


def test_d_optimal_ties_choose_smallest_concentration(monkeypatch):
    monkeypatch.setattr(DOptimalDesign, 'score_candidates', lambda self, state: (2., 2., 1.))
    assert DOptimalDesign(1, 1).select_next(DesignState(0.1, 10, 4, (), (3., 1., 2.))) == 1


def test_d_optimal_noise_scaling_rng_and_candidate_order():
    state = DesignState(0.05, 20, 6, (0.05, 1., 20.), (0.1, 1., 10., 20.))
    rng = make_rng(123)
    control = make_rng(123)
    selected = DOptimalDesign(1, 1).select_next(state, rng=rng)
    assert DOptimalDesign(1, 1, noise_std=0.05).select_next(state) == selected
    reversed_state = DesignState(0.05, 20, 6, state.selected_concentrations,
                                 tuple(reversed(state.candidate_concentrations)))
    assert DOptimalDesign(1, 1).select_next(reversed_state) == selected
    np.testing.assert_array_equal(rng.normal(size=10), control.normal(size=10))


def test_d_optimal_can_repeat_existing_point_and_rejects_complete_state():
    state = DesignState(0.1, 10, 4, (0.1, 10.), (10.,))
    assert DOptimalDesign(1, 1).select_next(state) == 10
    with pytest.raises(ValueError, match='complete'):
        DOptimalDesign(1, 1).select_next(DesignState(0.1, 10, 2, (0.1, 10.), (1.,)))


def test_d_optimal_updates_with_current_parameter_estimate():
    state = DesignState(0.05, 20, 8, (0.05, 1., 20.),
                        tuple(np.geomspace(0.05, 20, 200)))
    assert DOptimalDesign(0.2, 1).select_next(state) != DOptimalDesign(5, 1).select_next(state)


@pytest.mark.parametrize('kwargs', [{'km': 0}, {'vmax': -1}, {'noise_std': 0}])
def test_d_optimal_rejects_invalid_estimates(kwargs):
    values = dict(km=1, vmax=1, noise_std=1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        DOptimalDesign(**values)
