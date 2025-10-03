import struct


def read_u8(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x01])
    return struct.unpack("<B", data)[0]


def read_u16(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<H", data)[0]


def read_u32(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x04])
    return struct.unpack("<I", data)[0]


def read_s8(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x01])
    return struct.unpack("<b", data)[0]

def read_s16(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<h", data)[0]


def read_s32(data: bytes, addr) -> int:
    data = bytes(data[addr : addr + 0x04])
    return struct.unpack("<i", data)[0]


def read_f16(data: bytes, addr) -> float:
    data = bytes(data[addr : addr + 0x02])
    return struct.unpack("<f", data)[0]


def read_f32(data: bytes, addr) -> float:
    data = bytes(data[addr : addr + 0x04])
    return struct.unpack("<f", data)[0]


def read_fx16(data: bytes, addr) -> float:
    return read_s16(data, addr) / 0x1000  # bit shift 12 bits to the left


def read_fx32(data: bytes, addr) -> float:
    return read_s32(data, addr) / 0x1000  # bit shift 12 bits to the left


def read_vector_2d(data: bytes, addr, addr2=None) -> tuple[float, float]:
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    return x, y


def read_vector_3d(data: bytes, addr, addr2=None, addr3=None) -> tuple[float, float, float]:
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    z = read_fx32(data, addr + 0x08 if addr3 is None else addr3)
    return x, y, z

def read_vector_4d(data: bytes, addr, addr2=None, addr3=None, addr4=None) -> tuple[float, float, float, float]:
    x = read_fx32(data, addr)
    y = read_fx32(data, addr + 0x04 if addr2 is None else addr2)
    z = read_fx32(data, addr + 0x08 if addr3 is None else addr3)
    w = read_fx32(data, addr + 0x0C if addr4 is None else addr4)
    return x, y, z, w

def read_matrix_4d(data: bytes, addr) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float]
]:
    vec_0 = read_vector_4d(data, addr)
    vec_1 = read_vector_4d(data, addr + 0x10)
    vec_2 = read_vector_4d(data, addr + 0x20)
    vec_3 = read_vector_4d(data, addr + 0x30)
    return vec_0, vec_1, vec_2, vec_3
