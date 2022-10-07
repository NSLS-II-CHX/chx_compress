import io

import numpy as np
import sparse

import pytest

from chx_compress.io.multifile.multifile import (
    MultifileWriter,
    write_sparse_coo_to_multifile,
)


@pytest.fixture()
def dense_data_factory():
    """Return a factory function that builds a ndarray of random values.

    The returned array is meant to resemble CHX detector data.  It has
    shape FRAMES x ROWS x COLUMNS. By default some of the random values
    will be zero.
    """
    rng = np.random.default_rng()

    def _sparse_data_factory(
        frame_count, row_count, column_count, low_value=0, high_value=3
    ):
        # sparse_data is meant to resemble CHX detector data
        sparse_data = rng.integers(
            low=low_value,
            high=high_value,
            size=frame_count * row_count * column_count,
            dtype=np.uint32,
        )
        sparse_data = sparse_data.reshape((frame_count, row_count, column_count))

        return sparse_data

    return _sparse_data_factory


@pytest.fixture()
def header_info_factory():
    """ Return a factory function that builds a multifile header dictionary. """
    def _header_info_factory(byte_count, row_count, column_count):
        header_info = {
            "byte_count": byte_count,
            "beam_center_x": 1143.0,
            "beam_center_y": 1229.0,
            "count_time": 0.001997,
            "detector_distance": 16235.550295,
            "frame_time": 0.002,
            "incident_wavelength": 1.28507668359453,
            "x_pixel_size": 7.5e-05,
            "y_pixel_size": 7.5e-05,
            "nrows": row_count,
            "ncols": column_count,
            "rows_begin": 0,
            "rows_end": row_count,
            "cols_begin": 0,
            "cols_end": column_count,
        }
        return header_info

    return _header_info_factory


@pytest.fixture()
def multifile_bytes_factory(dense_data_factory, header_info_factory):
    """ Return a factory function that builds a multifile. """
    def _multifile_factory(
        byte_count, frame_count, row_count, column_count, low_value=0, high_value=3
    ):
        header_info = header_info_factory(byte_count, row_count, column_count)

        dense_data = dense_data_factory(
            frame_count, row_count, column_count, low_value, high_value
        )
        sparse_coo_data = sparse.COO(dense_data)

        with MultifileWriter(io.BytesIO(), **header_info) as multifile_writer:
            write_sparse_coo_to_multifile(
                sparse_coo_array=sparse_coo_data, multifile_writer=multifile_writer
            )

            multifile_bytes = multifile_writer._write_buffer.getvalue()

        return multifile_bytes, header_info, sparse_coo_data

    return _multifile_factory
