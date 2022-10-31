import itertools
from pathlib import Path

import pytest

import numpy as np
import sparse

from chx_compress.io.multifile.multifile_yg import Multifile
from chx_compress.io.multifile.multifile import (
    MultifileWriter,
    write_sparse_coo_to_multifile,
)

"""
The Multifile class under comparison does not support 0 or 1 frames.
"""
@pytest.mark.parametrize(
    "byte_count,frame_count", itertools.product((2, 4, 8), (2, 3))
)
def test_write_and_read_with_different_code(
    tmp_path, header_info_factory, dense_data_factory, byte_count, frame_count
):
    # we need a file
    multifile_path = str(tmp_path / "a_multifile.cmp")

    row_count = 3
    column_count = 4

    header_info = header_info_factory(
        byte_count=byte_count, row_count=row_count, column_count=column_count
    )

    dense_data = dense_data_factory(
        frame_count=frame_count,
        row_count=row_count,
        column_count=column_count,
    )
    sparse_coo_data = sparse.COO(dense_data)

    with MultifileWriter(open(multifile_path, "wb"), **header_info) as multifile_writer:
        write_sparse_coo_to_multifile(
            sparse_coo_array=sparse_coo_data, multifile_writer=multifile_writer
        )

    multifile = Multifile(
        filename=multifile_path,
        beg=0,
        end=frame_count,
        reverse=True,  # default value is False
    )

    # read all frames so the file can be closed
    multifile_frame_list = [multifile.rdframe(i) for i in range(frame_count)]

    multifile.FID.close()

    # nrows, ncols are swapped in multifile.md relative to header_info
    #assert multifile.md == header_info
    for multifile_frame_i, dense_data_i in zip(multifile_frame_list, dense_data):
        assert np.array_equal(multifile_frame_i, dense_data_i)
