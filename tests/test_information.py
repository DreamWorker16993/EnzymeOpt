import numpy as np
import pytest

from enzymeopt.information import fisher_information, information_logdet


def test_information_hand_calculation_and_additivity():
    # KM=1,Vmax=2: gradients at S=1,3 are (-1/2,1/2),(-3/8,3/4).
    expected = np.array([[25/64, -17/32], [-17/32, 13/16]]) / 4
    actual = fisher_information([1, 3], km=1, vmax=2, noise_std=2)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual, sum(
        fisher_information([s], km=1, vmax=2, noise_std=2) for s in [1, 3]))


def test_empty_and_repeated_information():
    np.testing.assert_array_equal(fisher_information([], km=1, vmax=1), np.zeros((2, 2)))
    single = fisher_information([1], km=1, vmax=1)
    np.testing.assert_allclose(fisher_information([1, 1], km=1, vmax=1), 2 * single)
    assert information_logdet(single) == -np.inf


@pytest.mark.parametrize('scale', [1e-200, 1, 1e200])
def test_logdet_stable_across_scales(scale):
    assert information_logdet(np.diag([2., 3.]) * scale) == pytest.approx(
        np.log(6.) + 2 * np.log(scale))


@pytest.mark.parametrize('sigma', [0, -1, np.inf, np.nan, True])
def test_invalid_noise(sigma):
    with pytest.raises(ValueError):
        fisher_information([1], km=1, vmax=1, noise_std=sigma)


@pytest.mark.parametrize('matrix', [[[1, 2], [0, 1]], [[1, 0], [0, -1]],
                                    [[np.nan, 0], [0, 1]], [1, 2]])
def test_invalid_information(matrix):
    with pytest.raises(ValueError):
        information_logdet(matrix)


def test_unresolved_information_is_reported_as_singular():
    assert information_logdet(np.diag([1., 1e-18])) == -np.inf
