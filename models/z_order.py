import numpy as np


def round_to_int_32(data):
    """
    Takes a Numpy array of float values between
    -1 and 1, and rounds them to significant
    32-bit integer values, to be used in the
    morton code computation

    :param data: multidimensional numpy array
    :return: same as data but in 32-bit int format
    """

    min_data = np.abs(np.min(data)-0.5)
    data = 256*(data + min_data)

    data = np.round(2 ** 21 - data).astype(dtype=np.int32)

    return data


def split_by_3(x):
    """
    Method to separate bits of a 32-bit integer
    by 3 positions apart, using the magic bits
    https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/

    :param x: 32-bit integer
    :return: x with bits separated
    """


    x &= 0x1fffff

    x = (x | (x << 32)) & 0x1f00000000ffff

    x = (x | (x << 16)) & 0x1f0000ff0000ff

    x = (x | (x << 8)) & 0x100f00f00f00f00f

    x = (x | (x << 4)) & 0x10c30c30c30c30c3

    x = (x | (x << 2)) & 0x1249249249249249

    return x


def get_z_order(x, y, z):
    """
    Given 3 arrays of corresponding x, y, z
    coordinates, compute the morton (or z) code for
    each point and return an index array
    We compute the Morton order as follows:
        1- Split all coordinates by 3 (add 2 zeros between bits)
        2- Shift bits left by 1 for y and 2 for z
        3- Interleave x, shifted y, and shifted z
    The mordon order is the final interleaved bit sequence

    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :return: index array with morton code
    """
    res = 0
    res |= split_by_3(x) | split_by_3(y) << 1 | split_by_3(z) << 2

    return res


def get_z_values(data):
    """
    Computes the z values for a point array
    :param data: Nx3 array of x, y, and z location

    :return: Nx1 array of z values
    """
    points_round = round_to_int_32(data)
    z = get_z_order(points_round[:, 0], points_round[:, 1], points_round[:, 2])

    return z
