import numpy as np

frames_numbers = np.arange(0, 200, dtype=np.int, step=5)

print(frames_numbers)

frames_numbers = ["{:08d}".format(n) for n in frames_numbers]

print(frames_numbers)
print(type(frames_numbers))