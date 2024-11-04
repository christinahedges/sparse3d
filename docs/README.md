<a href="https://github.com/christinahedges/sparse3d/actions/workflows/pytest.yml"><img src="https://github.com/christinahedges/sparse3d/workflows/pytest/badge.svg" alt="Test status"/></a>
<a href="https://github.com/christinahedges/sparse3d/actions/workflows/lint.yml"><img src="https://github.com/christinahedges/sparse3d/workflows/lint/badge.svg" alt="Lint status"/></a>
[![Documentation badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://christinahedges.github.io/sparse3d/)

# `sparse3d`

This package contains implimentations of a class for working with data that looks like a **single large sparse image** containing small regions of **dense data**. Really, this is designed for working with astronomical images.

## What is `sparse3D`?

We often have large images in astronomy that might look like the sketch below. In this sketch we have a large, sparse image with 4 dense regions; A, B, C, and D. In astronomy we have this situation often, where we have small images of point like stars spread over large images.

```
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
|   +-----+                      *    |
|                                     |
+-------------------------------------+
```

We may wish to calculate a model for star brightness in each of these regions. In this case, we likely do not care about the value of the model outside these regions, e.g. at the point in the image indicated by `*`. However, our model may still be based on where we are within this larger image.

Because of this, it is efficient for us to take this image and cast it into a sparse representation using [`scipy`'s](https://scipy.org/) `sparse` library.

Unfortunately `sparse` does not easily enable us to do this, as it only allows 2D arrays.

This small repository implements a way that we can hold the data corresponding to each of the sub images inside of a sparse array, by "unwrapping" the indices for the sub images and insetting them in the larger, sparse image.

## To Do

* Add an example using real data
* Update tests for ROISparse class, and add a notebook demonstration
  * Update the ROISparse class to ensure the outputs are 3D if dotted with a 1D array.
* Implement a `fit` function?
