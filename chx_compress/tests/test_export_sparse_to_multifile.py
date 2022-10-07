from io import BufferedRandom, BytesIO

import numpy as np
import sparse

import pytest


from chx_compress.io.multifile.multifile import MultifileReader


@pytest.mark.parametrize("byte_count", (2, 4, 8))
def test_sparse_to_multifile_export(multifile_bytes_factory, byte_count):
    """Export a sparse.COO array as a multifile.

    This is as much a demonstration as a test.
    """
    # build a frame_count x row_count x column_count ndarray
    #   similar to CHX detector data, where each array element
    #   holds the photon count for a detector element
    #   using np.uint32 as the data type for photon count
    frame_count = 2
    row_count = 3
    column_count = 5

    multifile_bytes, header_info, sparse_detector_data = multifile_bytes_factory(
        byte_count=byte_count,
        frame_count=frame_count,
        row_count=row_count,
        column_count=column_count,
    )

    # starting from a multifile, reconstruct a numpy array
    #   and compare it to the sparse.COO array
    recovered_detector_data = np.zeros((frame_count, row_count, column_count))
    with MultifileReader(
        BufferedRandom(BytesIO(initial_bytes=multifile_bytes))
    ) as multifile_reader:
        assert multifile_reader.version_info == b"Version-COMP0001"
        assert multifile_reader.header_info == header_info
        assert len(multifile_reader) == sparse_detector_data.shape[0]

        for image_i, (recovered_pixel_indices, recovered_pixel_values) in enumerate(
            multifile_reader
        ):
            sparse_image = sparse_detector_data[image_i]
            sparse_image_linear_indices = np.ravel_multi_index(
                sparse_image.coords, sparse_image.shape
            )
            assert np.array_equal(sparse_image_linear_indices, recovered_pixel_indices)
            assert np.array_equal(sparse_image.data, recovered_pixel_values)

            np.put(
                recovered_detector_data[image_i],
                recovered_pixel_indices,
                recovered_pixel_values,
            )
            # np.array_equal does not work with sparse arrays
            assert np.array_equal(
                sparse_image.todense(), recovered_detector_data[image_i]
            )

    assert np.array_equal(recovered_detector_data, sparse_detector_data.todense())
