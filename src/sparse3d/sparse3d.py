"""Classes for working with sparse data in three dimensions."""

# Standard library
from copy import deepcopy
from typing import List, Tuple

# Third-party
import numpy as np
from scipy import sparse

from .mixins import Sparse3DMathMixin

__all__ = ["Sparse3D", "ROISparse3D"]


class Sparse3D(Sparse3DMathMixin, sparse.coo_matrix):
    """Special class for working with stacks of sparse 3D images"""

    # def __init__(self, data, row, col, imshape):
    def __init__(
        self,
        data: np.ndarray,
        row: np.ndarray,
        col: np.ndarray,
        imshape: Tuple[int, int],
        imcorner: Tuple[int, int] = (0, 0),
    ) -> None:
        """
        Initialize a Sparse3D instance with 3D dense data.

        This class is designed enable work with small, dense sub images within a larger, sparse image.

        This class takes in 3D dense data, and the row and column positions within the larger image of each of the data points.
        The data, row and column should have shape (nrows, ncols, n sub images).

        For example, below represents 4 dense image sub images (A, B, C, D) within a larger, sparse image. This would be input using
        with data of shape (nrows, ncols, 4) corresponding to the size of the sub images in row and colum, and the number of sub images.
        The size of the larger image is specified with `imshape`. All sub images must have the same size.

        +-------------------------------------+
        |                                     |
        |   +-----+        +-----+            |
        |   |     |        |     |            |
        |   |  A  |        |  B  |            |
        |   |     |        |     |            |
        |   +-----+        +-----+            |
        |                                     |
        |                  +-----+            |
        |                  |     |            |
        |   +-----+        |  C  |            |
        |   |     |        |     |            |
        |   |  D  |        +-----+            |
        |   |     |                           |
        |   +-----+                           |
        |                                     |
        +-------------------------------------+


        Parameters
        ----------
        data : np.ndarray
            A dense 3D array containing data elements of shape `(nrows, ncols, n sub images)`.
            The shape of data defines the size and number of dense sub images.
        row : np.ndarray
            A 3D array indicating the row indices of non-zero elements, with shape `(nrows, ncols, n sub images)`.
        col : np.ndarray
            A 3D array indicating the column indices of non-zero elements, with shape `(nrows, ncols, n sub images)`.
        imshape : tuple of int
            A tuple `(row, column)` defining the shape of the larger, sparse image.
        imcorner : tuple of int
            A tuple `(row, column)` defining the corner of the larger, sparse image. Defaults to (0, 0)

        Raises
        ------
        ValueError
            If `data`, `row`, or `col` are not 3D arrays, or if their third dimensions do not match.

        """
        if not np.all([row.ndim == 3, col.ndim == 3, data.ndim == 3]):
            raise ValueError("Pass a 3D array (nrow, ncol, nsubimages)")
        self.nsubimages = data.shape[-1]
        self.imshape = imshape
        self.imcorner = imcorner

        if not np.all(
            [
                row.shape[-1] == self.nsubimages,
                col.shape[-1] == self.nsubimages,
            ]
        ):
            raise ValueError("Must have the same 3rd dimension (nsubimages).")
        self.subrow = row.astype(int) - self.imcorner[0]
        self.subcol = col.astype(int) - self.imcorner[1]
        self.subdepth = (
            np.arange(row.shape[-1], dtype=int)[None, None, :]
            * np.ones(row.shape, dtype=int)[:, :, None]
        )
        # The data for the sub images. We can not overwrite `self.data`, which is property of the COO matrix.
        self.subdata = data
        # We use this mask repeatedly so we calculate once on initialization.
        self._kz = self.subdata != 0

        self.subshape = row.shape
        self.cooshape = (np.prod([*self.imshape[:2]]), self.nsubimages)
        self.coord = (0, 0)
        super().__init__(self.cooshape)

        # In order to simulate the data being three dimensional, we unwrap the 3D indicies into 2D
        # (row, column) -> row position within the sparse array.
        index0 = (np.vstack(self.subrow)) * self.imshape[1] + (
            np.vstack(self.subcol)
        )
        # (nsubimage) -> column position in the sparse array.
        index1 = np.vstack(self.subdepth).ravel()

        # We will reuse this when we reset the data so we store it once.
        self._index_no_offset = np.vstack([index0.ravel(), index1.ravel()])
        # This mask represents where the input data is within bounds of self.imshape.
        self._submask_no_offset = np.vstack(
            self._get_submask(offset=(0, 0))
        ).ravel()

        # We use these to calculate translations, so we calculate them once.
        self._subrow_v = deepcopy(np.vstack(self.subrow).ravel())
        self._subcol_v = deepcopy(np.vstack(self.subcol).ravel())
        self._subdata_v = deepcopy(np.vstack(deepcopy(self.subdata)).ravel())

        self._index1 = np.vstack(self.subdepth).ravel()
        self._set_data()

    def multiply(self, other):
        """Returns a matrix with the same sparsity structure as self,
        but with different data. By default the index arrays are copied.
        """
        return self._new_s3d(
            new_data=self.subdata * other,
            new_row=self.subrow,
            new_col=self.subcol,
        )

    def __repr__(self):
        return f"<{(*self.imshape, self.nsubimages)} Sparse3D array of type {self.dtype}>"

    def __len__(self):
        return self.shape[-1]

    def tocoo(self):
        """Returns a COO matrix built from this Sparse3D instance."""
        return sparse.coo_matrix(
            (self.data, (self.row, self.col)), shape=self.cooshape
        )

    def copy(self):
        """Returns a deepcopy of self."""
        return deepcopy(self)

    def _new_s3d(self, new_data, new_row, new_col):
        """Convenience function to return a new version of this class"""
        return self.__class__(
            data=new_data, row=new_row, col=new_col, imshape=self.imshape
        )

    def __getitem__(self, index):
        """
        Handle indexing for Sparse3D. Only slices that are full-length for the
        first two dimensions and slicing on the last dimension are allowed.

        Parameters
        ----------
        index : tuple of slices
            A tuple representing the slicing operation (e.g., [:, :, 0] or [:, :, 1:3]).

        Returns
        -------
        Sparse3D
            A new Sparse3D instance containing the sliced data.

        Raises
        ------
        IndexError
            If the slicing does not span the full length of the first two dimensions or if indexing
            is attempted on the first two dimensions.
        """
        # Check if the index is a tuple and has three elements (for 3D slicing)
        if not isinstance(index, tuple) or len(index) != 3:
            raise IndexError(
                "Indexing must be for three dimensions (e.g., [:, :, 0])."
            )

        # Ensure the first two indices are full slices
        if index[0] != slice(None) or index[1] != slice(None):
            raise IndexError(
                "Only full slices (:) are allowed for the first two dimensions."
            )

        # Handle the third index (slicing along the last dimension)
        last_index = index[2]

        # Extract the relevant slice from the data
        if isinstance(last_index, int):
            # Return a 2D slice with only the specified last index
            new_data = self.subdata[:, :, last_index : last_index + 1]
            new_row = self.subrow[:, :, last_index : last_index + 1]
            new_col = self.subcol[:, :, last_index : last_index + 1]
        elif isinstance(last_index, slice):
            # Return a 3D slice with the range specified by the slice
            new_data = self.subdata[:, :, last_index]
            new_row = self.subrow[:, :, last_index]
            new_col = self.subcol[:, :, last_index]
        else:
            raise IndexError("The last index must be an integer or a slice.")

        # Create a new Sparse3D instance with the sliced data
        return self._new_s3d(
            new_data=new_data, new_row=new_row, new_col=new_col
        )

    def dot(self, other: np.ndarray) -> np.ndarray:
        """
        Compute the dot product with another array.

        This method calculates the dot product of this Sparse3D instance with a 1D or 2D `numpy.ndarray`. If `other`
        is a 1D array, it will be treated as a column vector for the dot product. The resulting product is reshaped
        to match the original image shape with an added dimension for multi-dimensional results if `other` is 2D.

        Parameters
        ----------
        other : np.ndarray
            A 1D or 2D array to perform the dot product with. The first dimension must be the number of sub images
            in the Sparse3D instance (i.e should match `self.nsubimages`). If the vector is 1D, it will be recast to have shape
            (n sub images, 1).

        Returns
        -------
        np.ndarray
            The resulting array from the dot product, reshaped to match the image dimensions `(n, *self.image_shape)`
            where n is the length of the second dimension of `other`.
            This will always be a 3D dataset.

        Raises
        ------
        NotImplementedError
            If `other` is not a `numpy.ndarray`.
        ValueError
            If the shape of `other` does not match the required dimensions for the dot product.
        """
        if isinstance(other, (int, float, list)):
            other = np.atleast_1d(other)
        if not isinstance(other, np.ndarray):
            raise NotImplementedError(
                f"dot products with type {type(other)} are not implemented."
            )
        if other.ndim == 1:
            result = super().dot(other).reshape(self.imshape)
        else:
            nt = other.shape[1]
            result = (
                super()
                .dot(other)
                .reshape((*self.imshape, nt))
                .transpose([2, 0, 1])
            )
        if hasattr(other, "unit"):
            return result * other.unit
        return result

    def _index(self, offset=(0, 0)):
        """
        Function gets the positions within the COO matrix (2D) of the input data (3D)
        If an offset is provided as a (row, column) tuple, this will also translate the
        input data by that amount.

        Parameters
        ----------
        offset: Tuple
            (row, column) tuple representing the offset to the row and column of each sub image.

        Returns
        -------
        index0 : np.ndarray
            An index with length self.imshape[0] * self.imshape[1] that describes where each
            element of the data should be placed in the COO matrix in the first index
        index1: np.ndarray
            An index with length self.nsubimages that describes where each
            element of the data should be placed in the COO matrix in the second index
        """
        if offset == (0, 0):
            return self._index_no_offset
        index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
            self._subcol_v + offset[1]
        )
        return index0, self._index1

    def _get_submask(self, offset=(0, 0)):
        """
        Hidden method to find where the data is within the array bounds. This mask can be used to ensure only
        data within the bounds of `self.imshape` is input to the sparse array when e.g. translated.

        Parameters
        ----------
        offset: Tuple
            (row, column) tuple representing the offset to the row and column of each sub image.

        Returns
        -------
        mask : np.ndarray
            A boolean mask with the same shape as input data, representing data points that are
            within bounds of self.imshape.
        """
        kr = ((self.subrow + offset[0]) < self.imshape[0]) & (
            (self.subrow + offset[0]) >= 0
        )
        kc = ((self.subcol + offset[1]) < self.imshape[1]) & (
            (self.subcol + offset[1]) >= 0
        )
        return kr & kc & self._kz

    def _set_data(self, offset=(0, 0)):
        """
        Hidden method to set the values into the correct position in the matrix.

        If (0, 0) is passed, this will reset the translation of the data.
        If any other tuple is passed, the row and column indexes of the input data will be
        translated by that much.

        This is an inplace operation.

        Parameters
        ----------
        offset: Tuple
            (row, column) tuple representing the offset to the row and column of each sub image.
        """
        if offset == (0, 0):
            index0, index1 = self._index((0, 0))
            self.row, self.col = (
                index0[self._submask_no_offset],
                index1[self._submask_no_offset],
            )
            self.data = self._subdata_v[self._submask_no_offset]
        else:
            # find where the data is within the array bounds
            k = self._get_submask(offset=offset)
            k = np.vstack(k).ravel()
            new_row, new_col = self._index(offset=offset)
            self.row, self.col = new_row[k], new_col[k]
            self.data = self._subdata_v[k]
        self.coord = offset

    def translate(self, offset: Tuple):
        """
        Translate the data in the array by `offset` in integer pixel positions.

        Position is a (row, column) tuple. If the user passes a position (1, 1) row and column indices of
        the data will be shifted by 1 pixel in the sparse image.

        This is an in place operation, and will move the data from the input positions. i.e. this translation does not stack.

        Parameters
        ----------
        offset: Tuple
            (row, column) tuple representing the offset to the row and column of each sub image.
        """
        self.reset()
        # If translating to (0, 0), do nothing
        if offset == (0, 0):
            return
        self.clear()
        self._set_data(offset=offset)
        return

    def reset(self):
        """Reset any translation back to the original data"""
        self._set_data(offset=(0, 0))
        self.coord = (0, 0)
        return

    def clear(self):
        """Clear data in the array. This function will remove all data in the array."""
        self.data = np.asarray([])
        self.row = np.asarray([])
        self.col = np.asarray([])
        self.coord = (0, 0)
        self.eliminate_zeros()
        return

    def to_ROISparse3D(
        self,
        nROIs: int,
        ROI_size: Tuple[int, int],
        ROI_corners: List[Tuple[int, int]],
    ) -> "ROISparse3D":
        return ROISparse3D(
            data=self.subdata,
            row=self.subrow,
            col=self.subcol,
            nROIs=nROIs,
            ROI_size=ROI_size,
            ROI_corners=ROI_corners,
            imshape=self.imshape,
        )


class ROISparse3D(Sparse3D):
    """Special version of a Sparse3D matrix which only populates and works with data within Regions of Interest."""

    def __init__(
        self,
        data: np.ndarray,
        row: np.ndarray,
        col: np.ndarray,
        imshape: Tuple[int, int],
        nROIs: int,
        ROI_size: Tuple[int, int],
        ROI_corners: List[Tuple[int, int]],
        imcorner: Tuple[int, int] = (0, 0),
    ) -> None:
        """
        Initialize a Sparse3D instance with 3D dense data, and specify regions of interest that are required by the user.

        This class is designed enable work with small, dense sub images within a larger, sparse image.

        In some applications, we care about "regions of interest" within a larger image. Some examples;

            1. Target Pixel Files from NASA Kepler are small regions of interest from a large image.
            Positions of the TPFs in the larger image dictate the background expected, or the PSF shape expected.
            Inside each Target Pixel File there might be many stars
            2. Similarly, NASA Pandora will downlink regions of interest from a larger full frame image. The
            regions of interet may contain several targets.

        This gives us an updated image below:

        +-------------------------------------------------+
        |                                                 |
        |   +-----------+          +-----------+          |
        |   |  +---+    |          |  +---+    |          |
        |   |  | A |    |          |  | B |    |          |
        |   |  +---+    |          |  +---+    |          |
        |   |  ROI 1    |          |  ROI 2  +---+        |
        |   +-----------+          +---------| E |        |
        |                                    +---+        |
        |                                                 |
        |                    +-----------+                |
        |                    |         +---+              |
        |                    |         | C |              |
        |   +-----------+    |         +---+              |
        |   |    +---+  |    |   ROI 3   |                |
        |   |    | D |  |    +-----------+                |
        |   |    +---+  |                                 |
        |   |   ROI 4   |                     +---+       |
        |   +-----------+                     | F |       |
        |                                     +---+       |
        +-------------------------------------------------+

        In this case, we want to understand the images and their relative position within a larger image,
        but we do not want to calculate any values outside of those regions of interest, and we want
        the data to be returned to us with the shape of the region of interest. We do not expect the user
        to provide data outside of the regions of interest. For example, in the above diagram the "F"
        sub image is not close to a region of interest, and so would be superfluous.

        Parameters
        ----------
        data : np.ndarray
            A dense 3D array containing data elements of shape `(nrows, ncols, n sub images)`.
            The shape of data defines the size and number of dense sub images.
        row : np.ndarray
            A 3D array indicating the row indices of non-zero elements, with shape `(nrows, ncols, n sub images)`.
        col : np.ndarray
            A 3D array indicating the column indices of non-zero elements, with shape `(nrows, ncols, n sub images)`.
        imshape : tuple of int
            A tuple `(row, column)` defining the shape of the larger, sparse image.
        nROIs: int
            The number of regions of interest in the larger image
        ROI_size: Tuple
            The size the regions of interest in (row, column) pixels. All ROIs must be the same size.
        ROI_corners: List[Tuple[int, int]]
            The origin (lower left) corner positon for each of the ROIs. Must have length nROIs.
        imcorner : tuple of int
            A tuple `(row, column)` defining the corner of the larger, sparse image. Defaults to (0, 0)

        Raises
        ------
        ValueError
            If `data`, `row`, or `col` are not 3D arrays, or if their third dimensions do not match.
        ValueError
            If corners are not passed for all ROIs
        ValueError
            If corners are not passed as tuples.
        """
        self.nROIs = nROIs
        self.ROI_size = ROI_size
        self.ROI_corners = ROI_corners
        # self.imcorner = imcorner
        self.get_ROI_mask = self._parse_ROIS(nROIs, ROI_size, ROI_corners)
        super().__init__(
            data=data, row=row, col=col, imshape=imshape, imcorner=imcorner
        )

    def _parse_ROIS(self, nROIs: int, ROI_size: tuple, ROI_corners: list):
        """Method checks the ROI inputs are allowable. Returns a function to obtain the boolean mask describing the ROIs"""
        if not len(ROI_corners) == nROIs:
            raise ValueError("Must pass corners for all ROIs.")
        if not np.all([isinstance(corner, tuple) for corner in ROI_corners]):
            raise ValueError("Pass corners as tuples.")

        def get_ROI_masks_func(row, column):
            mask = []
            for roi in range(nROIs):
                rmin, cmin = ROI_corners[roi]
                rmax, cmax = rmin + ROI_size[0], cmin + ROI_size[1]
                mask.append(
                    (row >= rmin)
                    & (row < rmax)
                    & (column >= cmin)
                    & (column < cmax)
                )
            return np.asarray(mask)

        return get_ROI_masks_func

    def __repr__(self):
        return f"<{(*self.imshape, self.nsubimages)} ROISparse3D array of type {self.dtype}, {self.nROIs} Regions of Interest>"

    def __len__(self):
        return self.shape[-1]

    # def _get_submask(self, offset=(0, 0)):
    #     # find where the data is within the array bounds
    #     kr = ((self.subrow + offset[0]) < self.imshape[0]) & (
    #         (self.subrow + offset[0]) >= 0
    #     )
    #     kc = ((self.subcol + offset[1]) < self.imshape[1]) & (
    #         (self.subcol + offset[1]) >= 0
    #     )
    #     # kroi = self.get_ROI_mask(self.subrow + offset[0], self.subcol + offset[0]).any(
    #     #     axis=0
    #     # )
    #     return kr & kc & self._kz  # & kroi

    def _new_s3d(self, new_data, new_row, new_col):
        """Convenience function to return a new version of this class"""
        return self.__class__(
            data=new_data,
            row=new_row,
            col=new_col,
            imshape=self.imshape,
            nROIs=self.nROIs,
            ROI_size=self.ROI_size,
            ROI_corners=self.ROI_corners,
        )

    def dot(self, other):
        """
        Compute the dot product with another array.

        This method calculates the dot product of this ROISparse3D instance with a 1D or 2D `numpy.ndarray` or sparse array
        If `other` is a 1D array, it will be treated as a column vector for the dot product. The resulting product is reshaped
        to match the original image shape with an added dimension for multi-dimensional results if `other` is 2D.

        Parameters
        ----------
        other : np.ndarray, sparse.csr_matrix
            A 1D or 2D array to perform the dot product with. The first dimension must be the number of sub images
            in the Sparse3D instance (i.e should match `self.nsubimages`). If the vector is 1D, it will be recast to have shape
            (n sub images, 1).

        Returns
        -------
        np.ndarray
            The resulting array from the dot product, reshaped to match the image dimensions `(self.nROIs, n, *self.ROI_size)`
            where n is the length of the second dimension of `other`.
            This will always be a 4D dataset.
        """
        ndim = other.ndim
        if isinstance(other, np.ndarray):
            other = sparse.csr_matrix(other).T
        if not sparse.issparse(other):
            raise ValueError("Must pass a `sparse` array to dot.")
        if not other.shape[0] == self.nsubimages:
            if other.shape[1] == self.nsubimages:
                other = other.T
            else:
                raise ValueError(
                    f"Must pass {(self.nsubimages, 1)} shape object."
                )
        sparse_array = super().tocsr().dot(other)

        R, C = np.meshgrid(
            np.arange(0, self.ROI_size[0]),
            np.arange(0, self.ROI_size[1]),
            indexing="ij",
        )
        array = np.zeros((self.nROIs, other.shape[1], *self.ROI_size))
        for rdx, c in enumerate(self.ROI_corners):
            idx = (R.ravel() + c[0] - self.imcorner[0]) * self.imshape[1] + (
                C.ravel() + c[1] - self.imcorner[1]
            )
            k = (idx >= 0) & (idx < self.shape[0])
            array[rdx, :, k.reshape(self.ROI_size)] = sparse_array[
                idx[k]
            ].toarray()  # ).reshape(self.ROI_size))
        if ndim == 1:
            return array[:, 0, :, :]
        return array

    def to_Sparse3D(self):
        return Sparse3D(
            data=self.subdata,
            row=self.subrow,
            col=self.subcol,
            imshape=self.imshape,
            imcorner=self.imcorner,
        )


def _stack_Sparse3d(arrays: List[Sparse3D]) -> Sparse3D:
    imshapes = set([ar.imshape for ar in arrays])
    if not len(imshapes) == 1:
        raise ValueError(
            "Can only stack `Sparse3D` instances with the same `imshape`."
        )

    def _stack(arrays):
        return np.vstack([ar.transpose([2, 0, 1]) for ar in arrays]).transpose(
            [1, 2, 0]
        )

    return Sparse3D(
        data=_stack([ar.subdata for ar in arrays]),
        row=_stack([ar.subrow + arrays[0].imcorner[0] for ar in arrays]),
        col=_stack([ar.subcol + arrays[0].imcorner[1] for ar in arrays]),
        imshape=arrays[0].imshape,
        imcorner=arrays[0].imcorner,
    )


def _stack_ROISparse3d(arrays: List[Sparse3D]) -> ROISparse3D:
    for checkname in ["imshape", "nROIs", "ROI_size"]:
        check = set([getattr(ar, checkname) for ar in arrays])
        if not len(check) == 1:
            raise ValueError(
                f"Can only stack `Sparse3D` instances with the same `{checkname}`."
            )

    corners0 = arrays[0].ROI_corners
    corners1 = [l for ar in arrays[1:] for l in ar.ROI_corners]
    check = set(list(corners0)) - set(list(corners1))
    if not len(check) == 0:
        raise ValueError(
            "Can only stack `Sparse3D` instances with the same `ROI_corner`'s."
        )

    def _stack(arrays):
        return np.vstack([ar.transpose([2, 0, 1]) for ar in arrays]).transpose(
            [1, 2, 0]
        )

    return ROISparse3D(
        data=_stack([ar.subdata for ar in arrays]),
        row=_stack([ar.subrow + arrays[0].imcorner[0] for ar in arrays]),
        col=_stack([ar.subcol + arrays[0].imcorner[1] for ar in arrays]),
        imshape=arrays[0].imshape,
        imcorner=arrays[0].imcorner,
        nROIs=arrays[0].nROIs,
        ROI_size=arrays[0].ROI_size,
        ROI_corners=arrays[0].ROI_corners,
    )


def stack(arrays: List[Sparse3D]):
    if np.all([isinstance(ar, ROISparse3D) for ar in arrays]):
        return _stack_ROISparse3d(arrays)
    elif np.all([isinstance(ar, Sparse3D) for ar in arrays]):
        return _stack_Sparse3d(arrays)
    else:
        raise ValueError("Input arrays must be the same data type.")
