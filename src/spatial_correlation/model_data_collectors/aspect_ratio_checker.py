import numpy as np

org = (450, 350)
resolutions = []
aspect_ratio = []

resolutions.append(org)
aspect_ratio.append(org[0] / org[1])
for i in reversed(np.arange(0.1, 1, 0.1)):
    print(np.round(i, 1))
    r = (org[0] * i, org[1] * i)
    resolutions.append(r)
    aspect_ratio.append(r[0] / r[1])

for i in range(len(resolutions)):
    print("res : {}, ratio: {}".format(resolutions[i], aspect_ratio[i]))


