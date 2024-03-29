import itertools

from io import BufferedRandom, BytesIO

import numpy as np
import pytest

from chx_compress.io.multifile.multifile import (
    MultifileReader,
    MultifileWriter,
)


@pytest.mark.parametrize("byte_count", (2, 4, 8))
def test_multifile_with_no_images(header_info_factory, byte_count):
    """
    A strange but testable situation.
    """
    header_info = header_info_factory(
        byte_count=byte_count, row_count=2070, column_count=2167
    )

    with MultifileWriter(BufferedRandom(BytesIO()), **header_info) as multifile_writer:

        # at this point the multifile header has been written

        with MultifileReader(multifile_writer._write_buffer) as multifile_reader:
            assert multifile_reader.version_info == b"Version-COMP0001"
            assert multifile_reader.header_info == header_info
            assert len(multifile_reader) == 0

            with pytest.raises(ValueError):
                multifile_reader[0]

            with pytest.raises(ValueError):
                multifile_reader.get_image_data(image_index=0)


# this test is parameterized to cover all allowed values for byte_count
#   and different numbers of images per multifile, including the case
#   of zero images which is unlikely but testable
@pytest.mark.parametrize(
    "byte_count,image_count", itertools.product((2, 4, 8), (0, 1, 2))
)
def test_write_read(header_info_factory, byte_count, image_count):
    """
    Write a multifile to a buffer, then read a multifile from that buffer.

    Parameters
    ----------
    byte_count : integer
        number of bytes used to store pixel values; only 2, 4, and 8 are allowed
    image_count : integer
        number of images to write into the multifile
    """

    # byte_count is the only value that changes in header_info
    header_info = header_info_factory(
        byte_count=byte_count, row_count=2070, column_count=2167
    )

    with MultifileWriter(BufferedRandom(BytesIO()), **header_info) as multifile_writer:

        written_data = []
        # write image_count images in the multifile
        for image_i in range(image_count):
            # pixel indices are 0, 1, 2, ...
            written_pixel_indices = np.arange(10 * image_i, dtype=np.uint32)
            written_pixel_values = np.ones(10 * image_i, dtype=np.uint32)
            multifile_writer.write_image(written_pixel_indices, written_pixel_values)

            # remember the written data so we can assert it is read correctly
            written_data.append((written_pixel_indices, written_pixel_values))

        multifile_writer._write_buffer.flush()
        the_data_we_wrote = multifile_writer._write_buffer.raw.getvalue()

        assert not multifile_writer._write_buffer.closed
    assert multifile_writer._write_buffer.closed

    # test context manager usage
    with MultifileReader(
        BufferedRandom(BytesIO(initial_bytes=the_data_we_wrote))
    ) as multifile_reader:
        assert multifile_reader.version_info == b"Version-COMP0001"
        assert multifile_reader.header_info == header_info
        assert len(multifile_reader) == image_count

        # test iteration
        image_array = np.zeros(
            (
                multifile_reader.header_info["nrows"],
                multifile_reader.header_info["ncols"],
            )
        )
        for image_i, (read_pixel_indices, read_pixel_values) in enumerate(
            multifile_reader
        ):
            written_pixel_indices, written_pixel_values = written_data[image_i]
            assert np.array_equal(written_pixel_indices, read_pixel_indices)
            assert np.array_equal(written_pixel_values, read_pixel_values)
            image_array[:] = 0
            np.put(image_array, read_pixel_indices, read_pixel_values)

        assert not multifile_reader._read_buffer.closed
    assert multifile_reader._read_buffer.closed

    # test direct usage
    multifile_reader = MultifileReader(
        BufferedRandom(BytesIO(initial_bytes=the_data_we_wrote))
    )
    assert multifile_reader.version_info == b"Version-COMP0001"
    assert multifile_reader.header_info == header_info
    assert len(multifile_reader) == image_count

    # test indexing
    for image_i in range(len(multifile_reader)):
        read_pixel_indices, read_pixel_values = multifile_reader[image_i]
        written_pixel_indices, written_pixel_values = written_data[image_i]
        assert np.array_equal(written_pixel_indices, read_pixel_indices)
        assert np.array_equal(written_pixel_values, read_pixel_values)

    assert not multifile_reader._read_buffer.closed
    multifile_reader.close()

    assert multifile_reader._read_buffer.closed
