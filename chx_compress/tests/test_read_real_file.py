import os

import pytest

import numpy as np

from chx_compress.io.multifile.multifile import get_dense_image, multifile_reader


@pytest.mark.skipif(
    not os.path.exists("/nsls2/data/chx/legacy/Compressed_Data"),
    reason="compressed data files are not available",
)
def test_read_a_real_file():
    """Read a real file.

    Compare results from MultifileReader and Multifile.
    """
    real_file_path = "/nsls2/data/chx/legacy/Compressed_Data/uid_0014eb14-9373-4a6c-915a-041f5cc6da96.cmp"
    
    with multifile_reader(real_file_path) as mfr:

        byte_count = mfr.header_info["byte_count"]
        assert byte_count == 4

        mfr_image_0 = np.zeros((mfr.header_info["ncols"], mfr.header_info["nrows"]))
        get_dense_image(mfr, image_index=0, target_image_array=mfr_image_0)
        mfr_image_199 = np.zeros((mfr.header_info["ncols"], mfr.header_info["nrows"]))
        get_dense_image(mfr, image_index=199, target_image_array=mfr_image_199)

    multifile = Multifile(real_file_path, beg=0, end=200)
    multifile_image_0 = multifile.rdframe(0)
    multifile_image_199 = multifile.rdframe(199)

    multifile.FID.close()

    assert np.array_equal(mfr_image_0.shape, multifile_image_0.shape)
    assert np.array_equal(mfr_image_0, multifile_image_0)

    assert np.array_equal(mfr_image_199.shape, multifile_image_199.shape)
    assert np.array_equal(mfr_image_199, multifile_image_199)
