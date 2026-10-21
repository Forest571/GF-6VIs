import numpy as np
import pandas as pd

from gf6vis.indices import calculate_indices, calculate_indices_from_bands


def test_calculate_indices_from_dataframe():
    data = pd.DataFrame(
        {
            "Blue": [0.1, 0.2],
            "Red": [0.2, 0.3],
            "NIR": [0.6, 0.9],
            "RE1": [0.4, 0.5],
            "RE2": [0.5, 0.7],
        }
    )

    result = calculate_indices(data, indices=["NDVI", "CSI", "NCSI"])

    expected_ndvi = (data["NIR"] - data["Red"]) / (data["NIR"] + data["Red"])
    expected_csi = ((data["NIR"] - data["RE1"]) / (data["NIR"] + data["RE1"])) * (
        data["Blue"] / data["RE1"]
    )
    expected_ncsi = expected_csi * (data["NIR"] / data["RE2"])

    assert np.allclose(result["NDVI"], expected_ndvi)
    assert np.allclose(result["CSI"], expected_csi)
    assert np.allclose(result["NCSI"], expected_ncsi)


def test_calculate_indices_from_arrays():
    bands = {
        "Blue": np.array([0.1]),
        "Red": np.array([0.2]),
        "NIR": np.array([0.6]),
        "RE1": np.array([0.4]),
        "RE2": np.array([0.5]),
    }

    result = calculate_indices_from_bands(bands, indices=["MTCI", "real_MTCI"])

    assert np.allclose(result["MTCI"], np.array([0.5]))
    assert np.allclose(result["real_MTCI"], np.array([20.406]))
