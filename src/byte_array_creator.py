import numpy as np


def get_byte_array(size):
    # decide array size (key = size in KB)
    array_size = {
        2500: (50, 50),
        5000: (100, 50),
        7500: (150, 50),
        10000: (100, 100)
    }

    # np.random.seed(10)
    random_data = np.random.randint(low=10, high=250, size=array_size[size], dtype=np.uint8)
    print(random_data)

    b_array = bytearray(random_data)
    print(b_array)


data_size = 2500  # in bytes
get_byte_array(data_size)
