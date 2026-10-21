import numpy as np
import pandas as pd

from gf6vis.analysis import combine_bands, correlation_and_r2


def test_combine_bands_ratio():
    data = pd.DataFrame({"B1": [1.0, 2.0], "B2": [2.0, 4.0], "Target": [3.0, 6.0]})

    result = combine_bands(data, ["B1", "B2"], mode="ratio")

    assert list(result.columns) == ["(B1/B2)", "(B2/B1)"]
    assert np.allclose(result["(B1/B2)"], [0.5, 0.5])
    assert np.allclose(result["(B2/B1)"], [2.0, 2.0])


def test_correlation_and_r2_perfect_linear():
    indices = pd.DataFrame({"NDVI": [1.0, 2.0, 3.0, 4.0]})
    target = pd.Series([2.0, 4.0, 6.0, 8.0], name="CAB")

    result = correlation_and_r2(indices, target, "CAB")

    assert result.loc[0, "Correlation"] == 1.0
    assert result.loc[0, "R2"] == 1.0
    assert result.loc[0, "N"] == 4
