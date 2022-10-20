import os

import pytest

import numpy as np

from chx_compress.io.multifile.multifile import multifile_reader


@pytest.mark.skipif(
    not os.path.exists("/nsls2/data/chx/legacy/Compressed_Data"),
    reason="compressed data files are not available",
)
def test_read_a_real_file():
    """Read a real file.

    This test may take up to 3 minutes.
    """
    with multifile_reader(
        "/nsls2/data/chx/legacy/Compressed_Data/uid_0014eb14-9373-4a6c-915a-041f5cc6da96.cmp",
    ) as mfr:

        byte_count = mfr.header_info["byte_count"]
        assert byte_count == 4

        # read the first image
        pixel_indices, pixel_values = mfr[0]
        assert pixel_values[0] == 1

        # read all the images
        image_array = np.zeros((mfr.header_info["nrows"], mfr.header_info["ncols"]))
        for pixel_indices, pixel_values in mfr:
            image_array[:] = 0
            np.put(image_array, pixel_indices, pixel_values)
