# Standard library
from typing import TYPE_CHECKING

# Third-party
import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    # First-party/Local
    from src.sparse3d import (
        Sparse3D,  # Replace with the actual path to Sparse3D
    )


class Sparse3DMathMixin:
    """Mixin class handles math operations for Sparse3D. Helps readability to have these in a mixin."""

    def _check_other_matrix_is_same_shape(self, other):
        if (
            (self.subcol != other.subcol).any()
            | (self.subrow != other.subrow).any()
            | (self.imshape != other.imshape)
            | (self.subshape != other.subshape)
        ):
            raise ValueError("Must have same base indicies.")

    def __mod__(self, other):
        """
        Element-wise modulo operation between this Sparse3D instance and another object.

        Parameters
        ----------
        other : Sparse3D or scalar
            The object to compute the modulo with this Sparse3D instance. If `other` is a Sparse3D,
            it must have matching shapes; if it is a scalar, each non-zero element is taken modulo `other`.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance with the result of the modulo operation.

        Raises
        ------
        ValueError
            If `other` is a Sparse3D instance but does not match in shape.
        ZeroDivisionError
            If modulo by zero is attempted.
        """
        if isinstance(other, self.__class__):
            self._check_other_matrix_is_same_shape(other)
            if np.any(other.data == 0):
                raise ZeroDivisionError(
                    "Modulo by zero encountered in Sparse3D modulo."
                )
            new_data = self.subdata % other.subdata
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )

        elif np.isscalar(other) | isinstance(other, np.ndarray):
            if other == 0:
                raise ZeroDivisionError("Modulo by zero is not allowed.")
            # Apply modulo operation to each non-zero element
            new_data = self.subdata % other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented

    def __rmod__(self, other):
        """Reverse modulo to handle scalar % Sparse3D isn't implemented"""
        raise NotImplementedError(
            "Modulo of a scalar with a Sparse3D is not implemented."
        )

    def __ge__(self, other: "Sparse3D") -> "Sparse3D":
        """
        Element-wise >= comparison between this Sparse3D instance and another object.

        Parameters
        ----------
        other : Sparse3D, scalar, or ndarray
            The object to compare to this Sparse3D instance. If `other` is a Sparse3D,
            it must have the same shape and non-zero structure. If `other` is a scalar,
            each element in `self.data` is compared to `other`.

        Returns
        -------
        Sparse3D
            A Sparse3D instance with boolean data representing where elements are equal.

        Raises
        ------
        ValueError
            If `other` is a Sparse3D instance but does not match in shape.
        """
        if isinstance(other, self.__class__):
            new_data = self.subdata > other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            new_data = self.subdata > other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented

    def __gt__(self, other: "Sparse3D") -> "Sparse3D":
        """
        Element-wise > comparison between this Sparse3D instance and another object.

        Parameters
        ----------
        other : Sparse3D, scalar, or ndarray
            The object to compare to this Sparse3D instance. If `other` is a Sparse3D,
            it must have the same shape and non-zero structure. If `other` is a scalar,
            each element in `self.data` is compared to `other`.

        Returns
        -------
        Sparse3D
            A Sparse3D instance with boolean data representing where elements are equal.

        Raises
        ------
        ValueError
            If `other` is a Sparse3D instance but does not match in shape.
        """
        if isinstance(other, self.__class__):
            new_data = self.subdata > other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            new_data = self.subdata > other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented

    def __le__(self, other: "Sparse3D") -> "Sparse3D":
        """
        Element-wise <= comparison between this Sparse3D instance and another object.

        Parameters
        ----------
        other : Sparse3D, scalar, or ndarray
            The object to compare to this Sparse3D instance. If `other` is a Sparse3D,
            it must have the same shape and non-zero structure. If `other` is a scalar,
            each element in `self.data` is compared to `other`.

        Returns
        -------
        Sparse3D
            A Sparse3D instance with boolean data representing where elements are equal.

        Raises
        ------
        ValueError
            If `other` is a Sparse3D instance but does not match in shape.
        """
        if isinstance(other, self.__class__):
            new_data = self.subdata <= other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            new_data = self.subdata <= other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented

    def __lt__(self, other: "Sparse3D") -> "Sparse3D":
        """
        Element-wise < comparison between this Sparse3D instance and another object.

        Parameters
        ----------
        other : Sparse3D, scalar, or ndarray
            The object to compare to this Sparse3D instance. If `other` is a Sparse3D,
            it must have the same shape and non-zero structure. If `other` is a scalar,
            each element in `self.data` is compared to `other`.

        Returns
        -------
        Sparse3D
            A Sparse3D instance with boolean data representing where elements are equal.

        Raises
        ------
        ValueError
            If `other` is a Sparse3D instance but does not match in shape.
        """
        if isinstance(other, self.__class__):
            new_data = self.subdata < other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            new_data = self.subdata < other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented

    def __eq__(self, other: "Sparse3D") -> "Sparse3D":
        """
        Element-wise equality comparison between this Sparse3D instance and another object.

        Parameters
        ----------
        other : Sparse3D, scalar, or ndarray
            The object to compare to this Sparse3D instance. If `other` is a Sparse3D,
            it must have the same shape and non-zero structure. If `other` is a scalar,
            each element in `self.data` is compared to `other`.

        Returns
        -------
        Sparse3D
            A Sparse3D instance with boolean data representing where elements are equal.

        Raises
        ------
        ValueError
            If `other` is a Sparse3D instance but does not match in shape.
        """
        if isinstance(other, self.__class__):
            new_data = self.subdata == other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            new_data = self.subdata == other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented
            return super(sparse.coo_matrix, self).__add__(other)

    def __ne__(self, other: "Sparse3D") -> "Sparse3D":
        """
        Element-wise inequality comparison between this Sparse3D instance and another object.

        Parameters
        ----------
        other : Sparse3D, scalar, or ndarray
            The object to compare to this Sparse3D instance. If `other` is a Sparse3D,
            it must have the same shape and non-zero structure. If `other` is a scalar,
            each element in `self.data` is compared to `other`.

        Returns
        -------
        Sparse3D
            A Sparse3D instance with boolean data representing where elements are not equal.

        Raises
        ------
        ValueError
            If `other` is a Sparse3D instance but does not match in shape.
        """
        if isinstance(other, self.__class__):
            new_data = self.subdata != other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            new_data = self.subdata != other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented
            return super(sparse.coo_matrix, self).__add__(other)

    def __add__(self, other: "Sparse3D") -> "Sparse3D":
        """
        Add a value to this Sparse3D instance.

        In the case that `other` is a Sparse3D, this method performs element-wise addition, returning a new Sparse3D instance
        with the resulting data. Both Sparse3D instances must have matching row, column, image shapes, and sub image shapes.

        Parameters
        ----------
        other : object
            Another object to add to this one.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance containing the sum of this instance and `other`.

        Raises
        ------
        ValueError
            If the `row`, `col`, `imshape`, or `subshape` attributes do not match between the instances.
        """
        if isinstance(other, self.__class__):
            data = self.subdata + other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            data = self.subdata + other
            return self.__class__(
                data=data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented
            return super(sparse.coo_matrix, self).__add__(other)

    def __mul__(self, other: "Sparse3D") -> "Sparse3D":
        """
        Elementwise multiplcation of a value to this Sparse3D instance.

        In the case that `other` is a Sparse3D, this method performs element-wise addition, returning a new Sparse3D instance
        with the resulting data. Both Sparse3D instances must have matching row, column, image shapes, and sub image shapes.

        Parameters
        ----------
        other : object
            Another object to multiply this one.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance containing the multiplication of this instance and `other`.

        Raises
        ------
        ValueError
            If the `row`, `col`, `imshape`, or `subshape` attributes do not match between the instances.
        """
        if isinstance(other, self.__class__):
            new_data = self.subdata * other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            new_data = self.subdata * other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented
            return super(sparse.coo_matrix, self).__add__(other)

    def __rmul__(self, other):
        """
        Reverse multiplication to handle scalar * Sparse3D.

        Parameters
        ----------
        other : scalar
            The scalar value to multiply with each non-zero element in this Sparse3D instance.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance with each non-zero element multiplied by `other`.
        """
        # Just call __mul__ to handle the operation
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Element-wise division between this Sparse3D instance and another object.

        Parameters
        ----------
        other : Sparse3D or scalar
            The object to divide with this Sparse3D instance. If `other` is a Sparse3D,
            it must have matching shapes; if it is a scalar, each non-zero element is divided by the scalar.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance with the resulting data from the division.

        Raises
        ------
        ValueError
            If `other` is a Sparse3D instance but does not match in shape.
        ZeroDivisionError
            If division by zero is attempted.
        """
        if isinstance(other, self.__class__):
            self._check_other_matrix_is_same_shape(other)
            if np.any(other.data == 0):
                raise ZeroDivisionError(
                    "Division by zero encountered in Sparse3D division."
                )
            new_data = self.subdata / other.subdata
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        elif np.isscalar(other) | isinstance(other, np.ndarray):
            if np.isscalar(other):
                if other == 0:
                    raise ZeroDivisionError("Division by zero is not allowed.")
            if isinstance(other, np.ndarray):
                if (other == 0).any():
                    raise ZeroDivisionError("Division by zero is not allowed.")
            # Divide each non-zero element by a scalar
            new_data = self.subdata / other
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            data = self.subdata - other.subdata
            self._check_other_matrix_is_same_shape(other)
            return self.__class__(
                data=data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        else:
            return self.__add__(-other)

    def __radd__(self, other):
        """
        Handle reverse addition to ensure commutativity with scalars.

        Parameters
        ----------
        other : scalar
            A scalar value to be added to this Sparse3D instance.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance with the scalar added to all non-zero elements.
        """
        # Just call __add__ to handle the operation
        return self.__add__(other)

    def __rsub__(self, other):
        """
        Handle reverse addition to ensure commutativity with scalars.

        Parameters
        ----------
        other : scalar
            A scalar value to be added to this Sparse3D instance.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance with the scalar added to all non-zero elements.
        """
        if np.isscalar(other) | isinstance(other, np.ndarray):
            print(other)
            # Subtract self.data from other, effectively computing `other - self`
            new_data = other - self.subdata
            return self.__class__(
                data=new_data,
                row=self.subrow,
                col=self.subcol,
                imshape=self.imshape,
            )
        return NotImplemented

    def __pow__(self, power):
        """
        Override the power operator to apply the power to non-zero data elements.

        Parameters
        ----------
        power : int or float
            The exponent to raise each non-zero element to.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance with each non-zero element raised to the given power.

        Raises
        ------
        ValueError
            If `power` is not a valid number for exponentiation.
        """
        if not isinstance(power, (int, float)):
            raise ValueError("Power must be an integer or float.")
        new = self.copy()
        new.data = self.data**power
        new.subdata = self.subdata**power
        return new

    def __rpow__(self):
        raise NotImplementedError(
            "Raising values to the power of Sparse3D instances is not implemented."
        )

    def __neg__(self):
        """
        Negate all non-zero elements in the Sparse3D instance.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance with all non-zero elements negated.
        """
        negated_data = (
            -self.subdata
        )  # Negate all non-zero values in the data array
        return self.__class__(
            data=negated_data,
            row=self.subrow,
            col=self.subcol,
            imshape=self.imshape,
        )

    def __abs__(self):
        """
        Take the absolute value of all non-zero elements in the Sparse3D instance.

        Returns
        -------
        Sparse3D
            A new Sparse3D instance with the absolute values of all non-zero elements.
        """
        abs_data = np.abs(
            self.subdata
        )  # Absolute value of all non-zero values in the data array
        return self.__class__(
            data=abs_data,
            row=self.subrow,
            col=self.subcol,
            imshape=self.imshape,
        )

    def __array__(self, dtype=None):
        """
        Prevent conversion of Sparse3D to a dense array.

        Raises
        ------
        TypeError
            Always raises an error to prevent conversion to a dense array.
        """
        raise TypeError(
            """Sparse3D instances cannot be converted to dense arrays, because they are likely to be too large for memory.
                            You can force this by converting to a `sparse.csr_matrix`."""
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            ufunc_map = {
                np.exp: np.exp,
                np.log: np.log,
                np.log10: np.log10,
                np.cos: np.cos,
                np.sin: np.sin,
                np.sqrt: np.sqrt,
                np.tan: np.tan,
                np.arctan: np.arctan,
                np.arcsin: np.arcsin,
                np.arccos: np.arccos,
                np.sinh: np.sinh,
                np.cosh: np.cosh,
                np.tanh: np.tanh,
            }

            if ufunc in ufunc_map:
                new = self.copy()
                func = ufunc_map[ufunc]
                new.data = func(self.data, **kwargs)
                new.subdata = func(self.subdata, **kwargs)
                return new

            for input_ in inputs:
                if input_ is not self:
                    other = input_
                    break
            else:
                return NotImplemented

            if ufunc == np.add:
                return self.__add__(other)
            elif ufunc == np.subtract:
                return self.__rsub__(other)
            elif ufunc == np.multiply:
                return self.__mul__(other)
            elif ufunc == np.divide:
                return self.__truediv__(other)
        # For other ufuncs, fall back to the default behavior
        return NotImplemented
