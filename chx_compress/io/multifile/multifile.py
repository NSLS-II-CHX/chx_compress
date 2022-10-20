import os
import struct
import time

import numpy as np


"""
    MultifileReader and MultifileWriter provide direct
    and context manager interfaces for reading and writing
    the Mark Sutton "multifile" format from and to generic
    buffers.

    multifile_reader() and multifile_writer() are
    convenience functions that build the corresponding
    objects for reading and writing files.

    direct usage:

        mfw = multifile_writer("path/to/a/file", **required_header_info)
        mfw.write_image(pixel_indices=[1, 3, 5], pixel_values=[1, 2, 1])
        mfw.close()

        mfr = multifile_reader("path/to/a/file")
        image_count = len(mfr)
        pixel_indices, pixel_values = mfr[0]
        mfr.close()

    context manager usage:

        with multifile_writer("path/to/a/file", **required_header_info) as mfw:
            mfw.write_image(pixel_indices=[1, 3, 5], pixel_values=[1, 2, 1])
        
        with multifile_reader("path/to/a/file") as mfr:
            image_array = np.zeros(mfr.header_info["nrows"], header_info["ncols"])
            for pixel_indices, pixel_values in mfr:
                image_array[:] = 0
                numpy.put(image_array, pixel_indices, pixel_values)


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
    def __init__(self, read_buffer):
        self._read_buffer = read_buffer
        self._image_offsets = None
        self._read_header_and_offsets()

    def _read_header_and_offsets(self):
        """Read header and image offsets.

        This information is read before images are accessed.

        This method is intended for interactive use
        as well as by __enter__.
        """
        self._read_header()
        self._find_image_offsets()

    def close(self):
        """Close the read buffer.

        This method is intended for interactive use
        as well as by __exit__.
        """
        self._read_buffer.close()

    def __len__(self):
        return len(self._image_offsets)

    def __getitem__(self, index):
        return self.get_image_data(image_index=index)

    def __iter__(self):
        for image_i, _ in enumerate(self._image_offsets):
            yield self.get_image_data(image_index=image_i)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _read_header(self):
        """Read the header data and store it in self.header_info."""
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
        """Find all image offsets in a multifile.

        This step must be completed before images can be read.
        """
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

    def get_image_data(self, image_index):
        """Read and return image data by image index.

        Parameters
        ----------
        image_index : integer
            a valid image index, 0 <= image_index < len(self)

        Returns
        -------
        pixel_indices : list of numbers
            list of "flat" pixel indices for one image
        pixel_values : list of numbers
            list of non-zero pixel values corresponding to pixel_indices
        """
        if len(self._image_offsets) == 0:
            raise ValueError("multifile has no images")
        elif 0 <= image_index < len(self._image_offsets):
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

    Returns
    -------
    a MultifileReader object which must be explicitly closed
    by calling MultifileReader.close()
    """
    mfr = MultifileReader(read_buffer=open(filepath, mode=mode))

    return mfr


def get_dense_image(multifile_reader, image_index, target_image_array):
    """A convenience function to build a dense image array.

    Parameters
    ----------
    multifile_reader : MultifileReader

    image_index : int
      index of the image to return

    target_image_array : ndarray
      dense array for image data, will be overwritten

    Return
    ------
    dense_array : ndarray
      dense array representation of the multifile data

     """
    target_image_array[:] = 0
    np.put(target_image_array, *multifile_reader[image_index])
    return target_image_array


def get_dense_array(multifile_reader):
    """A convenience function to build a dense array of image data from a MultifileReader.

    Parameters
    ----------
    multifile_reader : MultifileReader

    Return
    ------
    dense_array : ndarray
      dense array representation of the multifile data

    """

    frame_count = len(multifile_reader)
    row_count = multifile_reader.header_info["nrows"]
    column_count = multifile_reader.header_info["ncols"]

    dense_array = np.zeros((frame_count, row_count, column_count))

    for image_i, (image_pixel_indices, image_pixel_values) in enumerate(
        multifile_reader
    ):
        np.put(
            dense_array[image_i], image_pixel_indices, image_pixel_values
        )

    return dense_array


class MultifileWriter:
    def __init__(
        self,
        write_buffer,
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
        self._write_buffer = write_buffer
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
        self.write_header()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write_header(self):
        self._write_buffer.write(self._header_buffer)

    def close(self):
        self._write_buffer.close()

    def write_image(self, pixel_indices, pixel_values):
        """Write one image.

        Parameters
        ----------
        pixel_indices : list of integers
            list of "flat" pixel indices

        pixel_values : list of numbers
            list of pixel values corresponding to the pixel_indices

        reference: https://github.com/NSLS-II/pyCHX/blob/2b36ac312f5d67fac8b6b4f6b8658ab589dd6dc9/pyCHX/chx_compress.py#L326
        """
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

    Return
    ------
    a MultifileWriter object
    """

    return MultifileWriter(open(filepath, "wb"), **kwargs)


def write_sparse_coo_to_multifile(sparse_coo_array, multifile_writer):
    """
    Read image data from a sparse.COO array and write it to a multifile.

    Parameters
    ----------
      sparse_coo_array: sparse.COO
        FRAMESxROWSxCOLUMNS source array of detector data

      multifile_writer: MultifileWriter
        multifile target for detector data
    """
    for sparse_image in sparse_coo_array:
        sparse_image_linear_indices = np.ravel_multi_index(
            sparse_image.coords, sparse_image.shape
        )
        multifile_writer.write_image(sparse_image_linear_indices, sparse_image.data)


"""    Description:

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


     Header contains 1024 bytes version name, 'beam_center_x', 'beam_center_y',
        'count_time', 'detector_distance', 'frame_time', 'incident_wavelength',
        'x_pixel_size', 'y_pixel_size', bytes per pixel (either 2 or 4
        (Default)), Nrows, Ncols, Rows_Begin, Rows_End, Cols_Begin, Cols_End,



"""


# TODO : split into RO and RW classes
class MultifileAPS:
    """
    Re-write multifile from scratch.

    """

    HEADER_SIZE = 1024

    def __init__(self, filename, mode="rb", nbytes=2):
        """
        Prepare a file for reading or writing.
        mode : either 'rb' or 'wb'
        numimgs: num images
        """
        if mode != "rb" and mode != "wb":
            raise ValueError("Error, mode must be 'rb' or 'wb'" "got : {}".format(mode))
        self._filename = filename
        self._mode = mode

        self._nbytes = nbytes
        if nbytes == 2:
            self._dtype = "<i2"
        elif nbytes == 4:
            self._dtype = "<i4"

        # open the file descriptor
        # create a memmap
        if mode == "rb":
            # self._fd = np.memmap(filename, dtype='c')
            self._fd = open(filename, "rb")
        elif mode == "wb":
            self._fd = open(filename, "wb")
        # frame number currently on
        self.index()
        self.beg = 0
        self.end = self.Nframes - 1

        # these are only necessary for writing
        hdr = self._read_header(0)
        self._rows = int(hdr["rows"])
        self._cols = int(hdr["cols"])

    def rdframe(self, n):
        # read header then image
        pos, vals = self._read_raw(n)
        img = np.zeros((self._rows * self._cols,))
        img[pos] = vals
        return img.reshape((self._rows, self._cols))

    def rdrawframe(self, n):
        # read header then image
        return self._read_raw(n)

    def index(self):
        """Index the file by reading all frame_indexes.
        For faster later access.
        """
        print("Indexing file...")
        t1 = time.time()
        cur = 0
        file_bytes = len(self._fd)

        self.frame_indexes = list()
        while cur < file_bytes:
            self.frame_indexes.append(cur)
            # first get dlen, 4 bytes

            self._fd.seek(cur + 152, os.SEEK_SET)
            # dlen = np.frombuffer(self._fd[cur+152:cur+156], dtype="<u4")[0]
            dlen = np.fromfile(self._fd, dtype=np.uint32, count=1)[0]
            print("found {} bytes".format(dlen))
            # self.nbytes is number of bytes per val
            cur += 1024 + dlen * (4 + self._nbytes)
            # break

        self.Nframes = len(self.frame_indexes)
        t2 = time.time()
        print("Done. Took {} secs for {} frames".format(t2 - t1, self.Nframes))

    def _read_header(self, n):
        """Read header from current seek position."""
        if n > self.Nframes:
            raise KeyError(
                "Error, only {} frames, asked for {}".format(self.Nframes, n)
            )
        # read in bytes
        cur = self.frame_indexes[n]
        # header_raw = self._fd[cur:cur + self.HEADER_SIZE]
        header = dict()
        self._fd.seek(cur + 108, os.SEEK_SET)
        header["rows"] = np.fromfile(self._fd, dtype=self._dtype, count=1)[0]
        self._fd.seek(cur + 112, os.SEEK_SET)
        header["cols"] = np.fromfile(self._fd, dtype=self._dtype, count=1)[0]
        self._fd.seek(cur + 116, os.SEEK_SET)
        header["nbytes"] = np.fromfile(self._fd, dtype=self._dtype, count=1)[0]
        self._fd.seek(cur + 152, os.SEEK_SET)
        header["dlen"] = np.fromfile(self._fd, dtype=self._dtype, count=1)[0]

        self._dlen = header["dlen"]
        self._nbytes = header["nbytes"]

        return header

    def _read_raw(self, n):
        """Read from raw.
        Reads from current cursor in file.
        """
        if n > self.Nframes:
            raise KeyError(
                "Error, only {} frames, asked for {}".format(self.Nframes, n)
            )
        cur = self.frame_indexes[n] + 1024
        dlen = self._read_header(n)["dlen"]

        # pos = self._fd[cur: cur+dlen*4]
        self._fd.seek(cur, os.SEEK_SET)
        pos = np.fromfile(self._fd, dtype=np.uint32, count=dlen)
        cur += dlen * 4
        # pos = np.frombuffer(pos, dtype='<i4')

        # TODO: 2-> nbytes
        vals = np.fromfile(self._fd, dtype=self._dtype, count=dlen)
        # vals = self._fd[cur: cur+dlen*2]
        # not necessary
        cur += dlen * 2
        # vals = np.frombuffer(vals, dtype=self._dtype)

        return pos, vals

    def _write_header(self, dlen, rows, cols):
        """Write header at current position."""
        self._rows = rows
        self._cols = cols
        self._dlen = dlen
        # byte array
        header = np.zeros(self.HEADER_SIZE, dtype="c")
        # write the header dlen
        header[152:156] = np.array([dlen], dtype="<i4").tobytes()
        # rows
        header[108:112] = np.array([rows], dtype="<i4").tobytes()
        # colds
        header[112:116] = np.array([cols], dtype="<i4").tobytes()
        self._fd.write(header)

    def write_raw(self, pos, vals):
        """Write a raw set of values for the next chunk."""
        rows = self._rows
        cols = self._cols
        dlen = len(pos)
        self._write_header(dlen, rows, cols)
        # now write the pos and vals in series
        pos = pos.astype(self._dtype)
        vals = vals.astype(self._dtype)
        self._fd.write(pos)
        self._fd.write(vals)


# TODO : split into RO and RW classes
class MultifileBNL:
    """
    Re-write multifile from scratch.

    """

    HEADER_SIZE = 1024

    def __init__(self, filename, mode="rb", version=2):
        """
        Prepare a file for reading or writing.
        mode : either 'rb' or 'wb'

        version : int, optional
            version 1 is old bnl format
            version 2 is the new format
        """
        self._version = version
        if mode == "wb":
            raise ValueError("Write mode 'wb' not supported yet")

        if mode != "rb" and mode != "wb":
            raise ValueError("Error, mode must be 'rb' or 'wb'" "got : {}".format(mode))

        self._filename = filename
        self._mode = mode

        # open the file descriptor
        # create a memmap
        if mode == "rb":
            # self._fd = np.memmap(filename, dtype='c')
            self._fd = open(filename, "rb")
        elif mode == "wb":
            self._fd = open(filename, "wb")

        # these are only necessary for writing
        self.md = self._read_main_header()
        self._rows = int(self.md["nrows"])
        self._cols = int(self.md["ncols"])

        # some initialization stuff
        self.nbytes = self.md["bytes"]
        if self.nbytes == 2:
            self.valtype = np.uint16
        elif self.nbytes == 4:
            self.valtype = np.uint32
        elif self.nbytes == 8:
            self.valtype = np.float64

        # frame number currently on
        self.index()

    def __len__(self):
        return self.Nframes

    def index(self):
        """Index the file by reading all frame_indexes.
        For faster later access.
        """
        print("Indexing file...")
        t1 = time.time()
        cur = self.HEADER_SIZE
        file_bytes = os.path.getsize(self._filename)
        # file_bytes = len(self._fd)

        self.frame_indexes = list()
        while cur < file_bytes:
            self.frame_indexes.append(cur)
            # first get dlen, 4 bytes

            # dlen = np.frombuffer(self._fd[cur:cur+4], dtype="<u4")[0]
            self._fd.seek(cur, os.SEEK_SET)
            # dlen = np.frombuffer(self._fd[cur+152:cur+156], dtype="<u4")[0]
            dlen = np.fromfile(self._fd, dtype=np.uint32, count=1)[0]
            # print("found {} bytes".format(dlen))
            # self.nbytes is number of bytes per val
            cur += 4 + dlen * (4 + self.nbytes)
            # break

        self.Nframes = len(self.frame_indexes)
        t2 = time.time()
        print("Done. Took {} secs for {} frames".format(t2 - t1, self.Nframes))

    def _read_main_header(self):
        """Read header from current seek position.

        Extracting the header was written by Yugang Zhang. This is BNL's
        format.
        1024 byte header +
        4 byte dlen + (4 + nbytes)*dlen bytes
        etc...
        Format:
            unsigned int beam_center_x;
            unsigned int beam_center_y;
        """
        # read in bytes
        # header is always from zero
        # header_raw = self._fd[cur:cur + self.HEADER_SIZE]
        ms_keys = [
            "beam_center_x",
            "beam_center_y",
            "count_time",
            "detector_distance",
            "frame_time",
            "incident_wavelength",
            "x_pixel_size",
            "y_pixel_size",
            "bytes",
            "nrows",
            "ncols",
            "rows_begin",
            "rows_end",
            "cols_begin",
            "cols_end",
        ]

        self._fd.seek(0, os.SEEK_SET)
        br = self._fd.read(1024)
        # magic = struct.unpack('@16s', br[:16])
        md_temp = struct.unpack("@8d7I916x", br[16:])
        self.md = dict(zip(ms_keys, md_temp))
        return self.md

    def _read_raw(self, n):
        """Read from raw.
        Reads from current cursor in file.
        """
        if n > self.Nframes:
            raise KeyError(
                "Error, only {} frames, asked for {}".format(self.Nframes, n)
            )
        # dlen is 4 bytes
        cur = self.frame_indexes[n]
        # dlen = np.frombuffer(self._fd[cur:cur+4], dtype="<u4")[0]
        self._fd.seek(cur, os.SEEK_SET)
        dlen = np.fromfile(self._fd, dtype=np.uint32, count=1)[0]
        cur += 4

        # pos = self._fd[cur: cur+dlen*4]
        # pos = np.frombuffer(pos, dtype='<u4')
        # self._fd.seek(cur,os.SEEK_SET)
        pos = np.fromfile(self._fd, dtype=np.uint32, count=dlen)

        cur += dlen * 4
        # TODO: 2-> nbytes
        # vals = self._fd[cur: cur+dlen*self.nbytes]
        # vals = np.frombuffer(vals, dtype=self.valtype)
        # self._fd.seek(cur,os.SEEK_SET)
        vals = np.fromfile(self._fd, dtype=self.valtype, count=dlen)

        return pos, vals

    def rdframe(self, n):
        # read header then image
        pos, vals = self._read_raw(n)
        img = np.zeros((self._rows * self._cols,))
        img[pos] = vals
        # trying to retain backwards compatibility of the old file
        if self._version > 1:
            img = img.reshape((self._rows, self._cols))
        else:
            img = img.reshape((self._cols, self._rows))
        return img

    def rdrawframe(self, n):
        # read header then image
        return self._read_raw(n)


class MultifileBNLCustom(MultifileBNL):
    def __init__(self, filename, beg=0, end=None, **kwargs):
        super().__init__(filename, **kwargs)
        self.beg = beg
        if end is None:
            end = self.Nframes - 1
        self.end = end

    def rdframe(self, n):
        if n > self.end:
            raise IndexError("Index out of range")
        return super().rdframe(n - self.beg)

    def rdrawframe(self, n):
        return super().rdrawframe(n - self.beg)
