import struct

import numpy as np


"""
Description:

    This is code that Mark wrote to open the multifile format
    in compressed mode, translated to python.
    This seems to work for DALSA, FCCD and EIGER in compressed mode.
    It should be included in the respective detector.i files
    Currently, this refers to the compression mode being '6'
    Each file is image descriptor files chunked together as follows:
            Header (1024 bytes)
    |--------------IMG N begin--------------|
    |                   Dlen
    |---------------------------------------|
    |       Pixel positions (dlen*4 bytes   |
    |      (0 based indexing in file)       |
    |---------------------------------------|
    |    Pixel data(dlen*bytes bytes)       |
    |    (bytes is found in header          |
    |    at position 116)                   |
    |--------------IMG N end----------------|
    |--------------IMG N+1 begin------------|
    |----------------etc.....---------------|


     Header contains 1024 bytes storing
        version name,
        beam_center_x,
        beam_center_y,
        count_time,
        detector_distance,
        frame_time,
        incident_wavelength,
        x_pixel_size,
        y_pixel_size,
        bytes per pixel (either 2 or 4(Default)),
        Nrows,
        Ncols,
        Rows_Begin,
        Rows_End,
        Cols_Begin,
        Cols_End,
"""

class MultifileReader:
    def __init__(self, read_buffer_factory):
        self._read_buffer_factory = read_buffer_factory

    def __enter__(self):
        self._read_buffer = self._read_buffer_factory()
        self._read_header()
        self._find_image_offsets()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._read_buffer.close()

    def _read_header(self):
        self._read_buffer.seek(0)

        version_size = struct.calcsize("@16s")
        version_data = self._read_buffer.read(version_size)
        self.version_info = struct.unpack_from("@16s", version_data)[0]

        header_size = struct.calcsize("@8d7I916x")
        header_data = self._read_buffer.read(header_size)
        self._header_values = struct.unpack_from("@8d7I916x", header_data)

        header_keys = (
            "beam_center_x",
            "beam_center_y",
            "count_time",
            "detector_distance",
            "frame_time",
            "incident_wavelength",
            "x_pixel_size",
            "y_pixel_size",
            "byte_count",
            "nrows",
            "ncols",
            "rows_begin",
            "rows_end",
            "cols_begin",
            "cols_end",
        )

        self.header_info = dict(zip(header_keys, self._header_values))

    def _find_image_offsets(self):
        image_offsets = []
        # this is the starting point for the image offset search
        #   there is no image previous to this offset
        next_image_offset = 1024
        while True:
            self._read_buffer.seek(next_image_offset, 0)

            # read dlen, it is 4 bytes
            dlen_data = self._read_buffer.read(4)
            if len(dlen_data) == 0:
                # there is no image data at this offset
                #   we have reached the end of this file
                break
            else:
                # there is image data at this offset
                image_offsets.append(next_image_offset)

                # calculate the offset of the next image (if one exists)
                dlen = struct.unpack("I", dlen_data)[0]
                # image data consists of two arrays:
                #   array of pixel indices
                #   array of pixel values
                # we just want to calculate the offset to the start of the next dlen
                pixel_indices_size = dlen * 4
                pixel_values_size = dlen * self.header_info["byte_count"]

                # we don't know yet if there will be an image at this offset
                #   tune in to the next loop iteration to find out!
                next_image_offset = (
                    next_image_offset + 4 + pixel_indices_size + pixel_values_size
                )

        self._image_offsets = tuple(image_offsets)

    def __len__(self):
        return len(self._image_offsets)

    def __getitem__(self, index):
        return self.get_image_data(image_index=index)

    def get_image_data(self, image_index):
        if 0 <= image_index < len(self._image_offsets):
            self._read_buffer.seek(self._image_offsets[image_index])
            # read dlen
            dlen_data = self._read_buffer.read(4)
            dlen = struct.unpack("I", dlen_data)[0]
            # what is the size of the corresponding pixel index list and pixel value array
            pixel_indices_size = dlen * 4
            pixel_indices_data = self._read_buffer.read(pixel_indices_size)
            pixel_indices = struct.unpack(f"{dlen}I", pixel_indices_data)
            pixel_values_size = dlen * self.header_info["byte_count"]
            pixel_values_data = self._read_buffer.read(pixel_values_size)
            if self.header_info["byte_count"] == 2:
                unpack_format = f"{dlen}h"
            elif self.header_info["byte_count"] == 4:
                unpack_format = f"{dlen}i"
            else:
                unpack_format = f"{dlen}d"
            pixel_values = struct.unpack(unpack_format, pixel_values_data)

            return pixel_indices, pixel_values
        else:
            raise ValueError(
                f"image_index: {image_index} is not in the range [0, {len(self._image_offsets)})"
            )


def multifile_reader(filepath, mode="rb"):
    """A convenience function to create a MultifileReader that reads from a file.

    Parameters
    ----------
    filepath : str
        path for the multifile
    mode : str
        optional mode to open the file, default is "rb"

    """
    return MultifileReader(lambda: open(filepath, mode=mode))


class MultifileWriter:
    def __init__(
        self,
        write_buffer_factory,
        beam_center_x,
        beam_center_y,
        count_time,
        detector_distance,
        frame_time,
        incident_wavelength,
        x_pixel_size,
        y_pixel_size,
        byte_count,
        nrows,
        ncols,
        rows_begin,
        rows_end,
        cols_begin,
        cols_end,
        pixel_value_format=None,
    ):
        self._write_buffer_factory = write_buffer_factory
        self._header_buffer = bytearray(1024)

        # pixel data format is determined by "byte_count"
        # which may be 2, 4, or 8
        if pixel_value_format is None:
            if byte_count == 2:
                self.pixel_value_format = "h"
            elif byte_count == 4:
                self.pixel_value_format = "i"
            elif byte_count == 8:
                self.pixel_value_format = "d"
            else:
                raise ValueError(f"byte_count must be 2, 4, or 8, but is {byte_count}")
        else:
            self.pixel_value_format = pixel_value_format

        struct.pack_into("@16s", self._header_buffer, 0, b"Version-COMP0001")
        struct.pack_into(
            "@8d7I916x",
            self._header_buffer,
            struct.calcsize("@16s"),
            beam_center_x,
            beam_center_y,
            count_time,
            detector_distance,
            frame_time,
            incident_wavelength,
            x_pixel_size,
            y_pixel_size,
            byte_count,
            nrows,
            ncols,
            rows_begin,
            rows_end,
            cols_begin,
            cols_end,
        )

    def __enter__(self):
        self._write_buffer = self._write_buffer_factory()
        self._write_buffer.write(self._header_buffer)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._write_buffer.close()

    def write_image(self, pixel_indices, pixel_values):
        # reference: https://github.com/NSLS-II/pyCHX/blob/2b36ac312f5d67fac8b6b4f6b8658ab589dd6dc9/pyCHX/chx_compress.py#L326
        if len(pixel_indices) == len(pixel_values):
            data_length = len(pixel_indices)
            self._write_buffer.write(struct.pack("@I", data_length))
            self._write_buffer.write(struct.pack(f"@{data_length}i", *pixel_indices))
            self._write_buffer.write(
                struct.pack(f"{data_length}{self.pixel_value_format}", *pixel_values)
            )
        else:
            raise ValueError(
                f"len(pixel_indices)={len(pixel_indices)} and len(pixel_values)={len(pixel_values)} must not be different"
            )


def multifile_writer(filepath, **kwargs):
    """A convenience function to create a MultifileWriter that writes to a file.

    Parameters
    ----------
    filepath : str
        path for the multifile
    kwargs :
        intended for the multifile header data
    """

    def file_factory():
        return open(filepath, "wb")

    return MultifileWriter(write_buffer_factory=file_factory, **kwargs)
