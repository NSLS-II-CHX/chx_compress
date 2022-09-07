from io import BufferedRandom, BytesIO

import numpy as np
import sparse

import pytest


from chx_compress.io.multifile.multifile import (
    MultifileReader,
    MultifileWriter,
    write_sparse_coo_to_multifile,
)


@pytest.mark.parametrize("byte_count", (2, 4, 8))
def test_sparse_to_multifile_export(byte_count):
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

    # detector_data is meant to resemble CHX detector data
    rng = np.random.default_rng()
    detector_data = rng.integers(
        low=0, high=3, size=frame_count * row_count * column_count, dtype=np.uint32
    )
    detector_data = detector_data.reshape((frame_count, row_count, column_count))

    # create a sparse.COO version
    sparse_detector_data = sparse.COO(detector_data)

    # write a multifile
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

    # starting from a sparse.COO array, write a multifile
    with MultifileWriter(BufferedRandom(BytesIO()), **header_info) as multifile_writer:

        write_sparse_coo_to_multifile(
            sparse_coo_array=sparse_detector_data, multifile_writer=multifile_writer
        )

        multifile_writer._write_buffer.flush()
        the_data_we_wrote = multifile_writer._write_buffer.raw.getvalue()

    # starting from a multifile, reconstruct a numpy array
    #   and compare it to the sparse.COO array
    recovered_detector_data = np.zeros((frame_count, row_count, column_count))
    with MultifileReader(
        BufferedRandom(BytesIO(initial_bytes=the_data_we_wrote))
    ) as multifile_reader:
        assert multifile_reader.version_info == b"Version-COMP0001"
        assert multifile_reader.header_info == header_info
        assert len(multifile_reader) == sparse_detector_data.shape[0]

        recovered_image_array = np.zeros(
            (
                multifile_reader.header_info["nrows"],
                multifile_reader.header_info["ncols"],
            )
        )
        for image_i, (recovered_pixel_indices, recovered_pixel_values) in enumerate(
            multifile_reader
        ):
            sparse_image = sparse_detector_data[image_i]
            sparse_image_linear_indices = np.ravel_multi_index(
                sparse_image.coords, sparse_image.shape
            )
            assert np.array_equal(sparse_image_linear_indices, recovered_pixel_indices)
            assert np.array_equal(sparse_image.data, recovered_pixel_values)

            recovered_image_array[:] = 0
            np.put(
                recovered_image_array, recovered_pixel_indices, recovered_pixel_values
            )
            # np.array_equal does not work with sparse arrays
            assert np.array_equal(sparse_image.todense(), recovered_image_array)

            recovered_detector_data[image_i] = recovered_image_array

    assert np.array_equal(recovered_detector_data, sparse_detector_data.todense())
