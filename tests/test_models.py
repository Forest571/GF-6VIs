import numpy as np

from gf6vis.models import fit_best_model, predict_curve


def test_fit_best_model_selects_linear_for_linear_data():
    x = np.arange(1, 6, dtype=float)
    y = 3 * x + 2

    result = fit_best_model(x, y, models=["linear", "quadratic"])

    assert result.r2 == 1.0
    assert result.n == 5
    pred = predict_curve(result, np.array([6.0]))
    assert np.allclose(pred, [20.0])
