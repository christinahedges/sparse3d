# Third-party
import numpy as np
import pytest
from scipy import sparse

# First-party/Local
from sparse3d import ROISparse3D, Sparse3D


def test_sparse3d():
    R, C = np.meshgrid(
        np.arange(20, 25).astype(int),
        np.arange(10, 16).astype(int),
        indexing="ij",
    )
    R = R[:, :, None] * np.ones(10, dtype=int)[None, None, :]
    C = C[:, :, None] * np.ones(10, dtype=int)[None, None, :]
    data = np.ones_like(R).astype(float)

    sw = Sparse3D(data, R, C, (50, 50))
    assert sw.imshape == (50, 50)
    assert sw.shape == sw.cooshape == (2500, 10)
    assert sw.subshape == R.shape
    assert isinstance(sw, sparse.coo_matrix)
    assert len(sw.data) == 300
    assert sw.data.sum() == 300
    assert sw.dtype == float

    # Move data out of frame
    sw = Sparse3D(data, R + 50, C, (50, 50))
    assert len(sw.data) == 0
    # translate back into frame
    sw.translate((-50, 0))
    assert len(sw.data) == 300
    # reset it
    sw.reset()
    assert len(sw.data) == 0

    sw = Sparse3D(data, R + np.arange(10), C + np.arange(10), (50, 50))
    sw.translate((-1, 1))
    assert len(sw.data) == 300

    assert sw.dot(np.ones(10)).shape == (50, 50)
    assert isinstance(sw.dot(np.ones(10)), np.ndarray)
    assert sw.dot(np.ones(10)).sum() == 300


def test_roisparse3d():
    R, C = np.mgrid[:20, :20]
    R, C = (
        R + np.arange(2, 48, 5)[:, None, None],
        C + np.arange(2, 48, 5)[:, None, None],
    )
    data = np.random.normal(0, 1, size=R.shape) ** 0

    sw = ROISparse3D(
        data,
        R,
        C,
        imshape=(50, 50),
        nROIs=3,
        ROI_size=(10, 10),
        ROI_corners=[(0, 0), (10, 40), (40, 41)],
    )
    assert sw.imshape == (50, 50)
    assert sw.ROI_size == (10, 10)
    assert sw.shape == sw.cooshape == (2500, 20)
    assert sw.subshape == R.shape
    assert isinstance(sw, sparse.coo_matrix)
    #    assert len(sw.data) == 370
    #    assert sw.data.sum() == 370
    assert sw.dtype == float

    assert sw.dot(np.ones(20)).shape == (3, 1, 10, 10)
    assert isinstance(sw.dot(np.ones(20)), np.ndarray)
    assert np.prod(sw.tocsr().dot(np.ones(20)).shape) == np.prod(sw.imshape)

    # translate the data away everything should be zero:
    sw.translate((-150, -150))
    assert sw.dot(np.ones(20)).shape == (3, 1, 10, 10)
    assert isinstance(sw.dot(np.ones(20)), np.ndarray)
    assert sw.dot(np.ones(20)).sum() == 0
    assert sw.tocsr().dot(np.ones(20)).sum() == 0

    sw.reset()
    assert sw.dot(np.ones(20)).shape == (3, 1, 10, 10)
    assert isinstance(sw.dot(np.ones(20)), np.ndarray)
    assert sw.dot(np.ones(20)).sum() != 0
    assert sw.tocsr().dot(np.ones(20)).sum() != 0


@pytest.fixture
def sw():
    R, C = np.meshgrid(
        np.arange(20, 25).astype(int),
        np.arange(10, 16).astype(int),
        indexing="ij",
    )
    R = R[:, :, None] * np.ones(10, dtype=int)[None, None, :]
    C = C[:, :, None] * np.ones(10, dtype=int)[None, None, :]
    data = np.ones_like(R).astype(float)

    return Sparse3D(data, R, C, (50, 50))


def test_add(sw):
    # Add a scalar
    for other_data in [
        2,
        np.random.normal(size=sw.nsubimages),
        np.random.normal(size=sw.subshape),
    ]:
        result = sw + other_data
        assert isinstance(result, Sparse3D)
        assert np.all(result.subdata == sw.subdata + other_data)

    # Add another Sparse3D with matching shape
    other = Sparse3D(
        data=sw.subdata.copy(),
        row=sw.subrow,
        col=sw.subcol,
        imshape=sw.imshape,
    )
    result = sw + other
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == sw.subdata + other.subdata)


def test_radd(sw):
    # Reverse addition with a scalar
    for other_data in [
        2,
        np.random.normal(size=sw.nsubimages),
        np.random.normal(size=sw.subshape),
    ]:
        result = other_data + sw
        assert isinstance(result, Sparse3D)
        assert np.all(result.subdata == sw.subdata + other_data)

    result = 2 * np.ones(sw.nsubimages) + sw
    assert isinstance(result, Sparse3D)
    assert np.all(result.data == sw.data + 2)


def test_sub(sw):
    # Subtract a scalar
    for other_data in [
        2,
        np.random.normal(size=sw.nsubimages),
        np.random.normal(size=sw.subshape),
    ]:
        result = sw - other_data
        assert isinstance(result, Sparse3D)
        assert np.all(result.subdata == sw.subdata - other_data)

    # Subtract another Sparse3D with matching shape
    other = Sparse3D(
        data=sw.subdata.copy(),
        row=sw.subrow,
        col=sw.subcol,
        imshape=sw.imshape,
    )
    result = sw - other
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == sw.subdata - other.subdata)


def test_rsub(sw):
    # Reverse subtraction with a scalar
    for other_data in [
        2,
        np.random.normal(size=sw.nsubimages),
        np.random.normal(size=sw.subshape),
    ]:
        result = other_data - sw
        assert isinstance(result, Sparse3D)
        assert np.all(result.subdata == other_data - sw.subdata)


def test_mul(sw):
    # Multiply by a scalar
    for other_data in [
        2,
        np.random.normal(size=sw.nsubimages),
        np.random.normal(size=sw.subshape),
    ]:
        result = sw * other_data
        assert isinstance(result, Sparse3D)
        assert np.all(result.subdata == sw.subdata * other_data)

    # Multiply by another Sparse3D with matching shape
    other = Sparse3D(
        data=sw.subdata.copy(),
        row=sw.subrow,
        col=sw.subcol,
        imshape=sw.imshape,
    )
    assert isinstance(sw, Sparse3D)
    assert isinstance(other, Sparse3D)
    result = sw * other
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == sw.subdata * other.subdata)


def test_rmul(sw):
    # Reverse multiplication with a scalar
    for other_data in [
        2,
        np.random.normal(size=sw.nsubimages),
        np.random.normal(size=sw.subshape),
    ]:
        result = other_data * sw
        assert isinstance(result, Sparse3D)
        assert np.all(result.subdata == other_data * sw.subdata)


def test_div(sw):
    # Divide by a scalar
    for other_data in [
        2,
        np.random.normal(size=sw.nsubimages) + 100,
        np.random.normal(size=sw.subshape) + 100,
    ]:
        result = sw / other_data
        assert isinstance(result, Sparse3D)
        assert np.all(result.subdata == sw.subdata / other_data)

    # Divide by another Sparse3D with matching shape
    other = Sparse3D(
        data=sw.subdata.copy(),
        row=sw.subrow,
        col=sw.subcol,
        imshape=sw.imshape,
    )
    result = sw / other
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == sw.subdata / other.subdata)


def test_power(sw):
    # Test raising elements to a power
    result = sw**2
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == sw.subdata**2)
    result = sw**0
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == sw.subdata**0)


def test_mod(sw):
    # Test modulo operation with scalar
    result = sw % 2
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == sw.subdata % 2)


def test_eq(sw):
    # Test equality with scalar
    result = sw == 2
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == (sw.subdata == 2))

    # Test equality with another Sparse3D with matching data
    other = Sparse3D(
        data=sw.subdata.copy(),
        row=sw.subrow,
        col=sw.subcol,
        imshape=sw.imshape,
    )
    result = sw == other
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata)


def test_ne(sw):
    # Test inequality with scalar
    result = sw != 2
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == (sw.subdata != 2))

    # Test inequality with another Sparse3D with different data
    other_data = sw.subdata * 0.4  # Modify data to test inequality
    other = Sparse3D(
        data=other_data, row=sw.subrow, col=sw.subcol, imshape=sw.imshape
    )
    result = sw != other
    assert isinstance(result, Sparse3D)
    assert np.any(result.subdata)  # At least one element should be True


def test_le(sw):
    # Test inequality with scalar
    result = sw <= 2
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == (sw.subdata <= 2))


def test_lt(sw):
    # Test inequality with scalar
    result = sw < 2
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == (sw.subdata < 2))


def test_ge(sw):
    # Test inequality with scalar
    result = sw >= 2
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == (sw.subdata >= 2))


def test_gt(sw):
    # Test inequality with scalar
    result = sw > 2
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == (sw.subdata > 2))


def test_neg(sw):
    # Test negation
    result = -sw
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == -sw.subdata)


def test_abs(sw):
    # Test absolute values
    result = abs(sw)
    assert isinstance(result, Sparse3D)
    assert np.all(result.subdata == np.abs(sw.subdata))


def test_np_exp(sw):
    result = np.exp(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.exp(sw.data)
    ), "Data should match np.exp result"


def test_np_log(sw):
    result = np.log(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.log(sw.data)
    ), "Data should match np.log result"


def test_np_log10(sw):
    result = np.log10(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.log10(sw.data)
    ), "Data should match np.log10 result"


def test_np_cos(sw):
    result = np.cos(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.cos(sw.data)
    ), "Data should match np.cos result"


def test_np_sin(sw):
    result = np.sin(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.sin(sw.data)
    ), "Data should match np.sin result"


def test_np_sqrt(sw):
    result = np.sqrt(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.sqrt(sw.data)
    ), "Data should match np.sqrt result"


def test_np_tan(sw):
    result = np.tan(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.tan(sw.data)
    ), "Data should match np.tan result"


def test_np_sinh(sw):
    result = np.sinh(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.sinh(sw.data)
    ), "Data should match np.sinh result"


def test_np_cosh(sw):
    result = np.cosh(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.cosh(sw.data)
    ), "Data should match np.cosh result"


def test_np_tanh(sw):
    result = np.tanh(sw)
    assert isinstance(result, Sparse3D), "Result should be a Sparse3D instance"
    assert np.allclose(
        result.data, np.tanh(sw.data)
    ), "Data should match np.tanh result"


def test_slice_last_dimension(sw):
    # Test slicing with a single index in the last dimension
    sliced_sw = sw[:, :, 1]
    assert isinstance(
        sliced_sw, Sparse3D
    ), "Result should be a Sparse3D instance"
    assert sliced_sw.subdata.shape == (
        5,
        6,
        1,
    ), "Sliced data should have a single slice in the last dimension"
    assert np.array_equal(
        sliced_sw.subdata, sw.subdata[:, :, 1:2]
    ), "Sliced data should match the expected slice"

    # Test slicing with a range in the last dimension
    sliced_sw = sw[:, :, 1:3]
    assert isinstance(
        sliced_sw, Sparse3D
    ), "Result should be a Sparse3D instance"
    assert sliced_sw.subdata.shape == (
        5,
        6,
        2,
    ), "Sliced data should have two slices in the last dimension"
    assert np.array_equal(
        sliced_sw.subdata, sw.subdata[:, :, 1:3]
    ), "Sliced data should match the expected range"


def test_invalid_slices(sw):
    # Test that slicing on the first two dimensions raises an error
    with pytest.raises(
        IndexError,
        match="Only full slices \\(:\\) are allowed for the first two dimensions.",
    ):
        sw[0, :, 1]

    with pytest.raises(
        IndexError,
        match="Only full slices \\(:\\) are allowed for the first two dimensions.",
    ):
        sw[:, 1, :]

    with pytest.raises(
        IndexError, match="Indexing must be for three dimensions"
    ):
        sw[:, :]

    with pytest.raises(
        IndexError, match="Indexing must be for three dimensions"
    ):
        sw[1]
