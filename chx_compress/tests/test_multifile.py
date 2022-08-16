import itertools
import os

from io import BufferedRandom, BytesIO

import numpy as np
import pytest

from chx_compress.io.multifile import MultifileReader, MultifileWriter


# this test is parameterized to cover all allowed values for byte_count
#   and different numbers of images per multifile
@pytest.mark.parametrize(
    "byte_count,image_count", itertools.product((2, 4, 8), (0, 1, 2))
)
def test_write_read(byte_count, image_count):
    """
    Write a multifile to a buffer, then read a multifile from that buffer.

    Parameters
    ----------
    byte_count : integer
        number of bytes used to store pixel values, only 2, 4, and 8 are allowed
    image_count : integer
        number of images to write into the multifile
    """

    # byte_count is the only value that changes in header_info
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
        "nrows": 2070,
        "ncols": 2167,
        "rows_begin": 0,
        "rows_end": 2070,
        "cols_begin": 0,
        "cols_end": 2167,
    }

    with MultifileWriter(
        lambda: BufferedRandom(BytesIO()), **header_info
    ) as multifile_writer:

        written_data = []
        # write image_count images in the multifile
        for image_i in range(image_count):
            written_pixel_indices = np.arange(10 * image_i, dtype=np.uint32)
            written_pixel_values = np.ones(10 * image_i, dtype=np.uint32)
            multifile_writer.write_image(written_pixel_indices, written_pixel_values)

            # remember the written data so we can assert it is read correctly
            written_data.append((written_pixel_indices, written_pixel_values))

        multifile_reader = MultifileReader(multifile_writer._write_buffer)
        assert multifile_reader.header_info == header_info
        assert len(multifile_reader) == image_count
        # read image_count images from the multifile
        for image_i in range(image_count):
            read_pixel_indices, read_pixel_values = multifile_reader[image_i]
            written_pixel_indices, written_pixel_values = written_data[image_i]
            assert np.array_equal(written_pixel_indices, read_pixel_indices)
            assert np.array_equal(written_pixel_values, read_pixel_values)


@pytest.mark.skipif(
    not os.path.exists("/nsls2/data/chx/legacy/Compressed_Data"),
    reason="compressed data files are not available",
)
def test_read_a_real_file():
    with open(
        "/nsls2/data/chx/legacy/Compressed_Data/uid_0014eb14-9373-4a6c-915a-041f5cc6da96.cmp",
        "rb",
    ) as f:
        multifile_reader = MultifileReader(read_buffer=f)

        print(multifile_reader.version_info)
        print(multifile_reader.header_info)

        byte_count = multifile_reader.header_info["byte_count"]
        print(f"byte_count: {byte_count}")

        # read the first image
        pixel_indices, pixel_values = multifile_reader.get_image(0)
        assert pixel_values[0] == 1
