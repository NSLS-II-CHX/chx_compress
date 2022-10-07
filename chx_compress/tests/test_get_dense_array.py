import io

import numpy as np

import pytest

from chx_compress.io.multifile.multifile import get_dense_array, MultifileReader


@pytest.mark.parametrize("byte_count", (2, 4, 8))
def test_get_dense_array(multifile_bytes_factory, byte_count):
    multifile_bytes, _, sparse_data = multifile_bytes_factory(
        byte_count=byte_count, frame_count=10, row_count=20, column_count=30
    )

    with MultifileReader(
        read_buffer=io.BufferedRandom(io.BytesIO(initial_bytes=multifile_bytes))
    ) as multifile_reader:
        dense_data = get_dense_array(multifile_reader=multifile_reader)

        assert np.array_equal(dense_data, sparse_data.todense())
