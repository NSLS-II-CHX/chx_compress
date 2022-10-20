import io

import numpy as np

import pytest

from chx_compress.io.multifile.multifile import get_dense_image, MultifileReader


@pytest.mark.parametrize("byte_count", (2, 4, 8))
def test_get_dense_array(multifile_bytes_factory, byte_count):
    frame_count = 10

    multifile_bytes, _, sparse_data = multifile_bytes_factory(
        byte_count=byte_count, frame_count=frame_count, row_count=20, column_count=30
    )

    with MultifileReader(
        read_buffer=io.BufferedRandom(io.BytesIO(initial_bytes=multifile_bytes))
    ) as multifile_reader:

        nrows = multifile_reader.header_info["nrows"]
        ncols = multifile_reader.header_info["ncols"]
        image_array = np.zeros((nrows, ncols))

        for frame_i in range(frame_count):
            get_dense_image(
                multifile_reader=multifile_reader,
                image_index=frame_i,
                target_image_array=image_array,
            )

            assert np.array_equal(image_array, sparse_data[frame_i].todense())
