"""
module to create mixed resoliution images using the overlap and given resolution. Counts the # pixels in resulting image
"""

import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def get_energy_consumption(pixel_count):
    """
    method to compute image capture energy consumption (mJ) for given count of pixels
    :param pixel_count:
    :return:
    """
    poly_features = PolynomialFeatures(degree=1, interaction_only=False)
    pixel_count = np.array(pixel_count).reshape(-1, 1)
    pixel_count = poly_features.fit_transform(pixel_count)
    # print(pixel_count)
    # load regression model
    model = None
    model_path = "capture_frame_energy_model"
    with open(model_path, 'rb') as in_file:
        model = pickle.load(in_file)
    assert model is not None

    # predict energy consumption
    energy_consumption = model.predict(pixel_count)
    return np.round(energy_consumption[0], decimals=3)


if __name__ == "__main__":
    ORG_RES = (1920, 1080)
    shared_reg_key = "WT_5_7"

    SHARED_REG_RES = (70, 70)
    NON_SHARED_REG_RES = (512, 512)
    shared_regions = {
        "PETS_5_7": [91, 142, 693, 510],
        "PETS_8_5": [267, 85, 716, 537],
        "WT_1_4": [6, 4, 908, 1074],
        "WT_5_7": [51, 139, 1507, 1041]
    }
    org_pixel_count = ORG_RES[0] * ORG_RES[1]
    sh_reg = shared_regions[shared_reg_key]
    sh_reg_width = sh_reg[2] - sh_reg[0]
    sh_reg_height = sh_reg[3] - sh_reg[1]

    # shared region pixel count
    sh_pixel_count = sh_reg_width * sh_reg_height
    sh_pixel_fraction = sh_pixel_count / org_pixel_count
    sh_pixel_count_new = int(SHARED_REG_RES[0] * SHARED_REG_RES[1] * sh_pixel_fraction)

    # sh_reg_new_width = (sh_reg_width / ORG_RES[0]) * SHARED_REG_RES[0]
    # sh_reg_new_width = int(sh_reg_new_width)
    # sh_reg_new_height = (sh_reg_height / ORG_RES[1]) * SHARED_REG_RES[1]
    # sh_reg_new_height = int(sh_reg_new_height)
    # print("Shared region (w,h): {}, {}".format(sh_reg_new_width, sh_reg_new_height))

    # non-shared region
    nsh_pixel_count = org_pixel_count - (sh_reg_width * sh_reg_height)
    nsh_pixel_fraction = nsh_pixel_count / org_pixel_count
    nsh_pixel_count_new = int(NON_SHARED_REG_RES[0] * NON_SHARED_REG_RES[1] * nsh_pixel_fraction)

    # total pixel count
    total_pixel_count = sh_pixel_count_new + nsh_pixel_count_new
    print("Org Pixel Count : {}".format(org_pixel_count))
    print("Shared Region Pixel Count : {}".format(sh_pixel_count_new))
    print("Non Shared Region pixel count : {}".format(nsh_pixel_count_new))
    print("new total pixel count : {}\n".format(total_pixel_count))

    # calculate energy savings
    print("Energy Savings!!.....")
    initial_energy_consumption = get_energy_consumption(org_pixel_count)
    print("Initial Energy : {}".format(initial_energy_consumption))
    collab_energy_consumption = get_energy_consumption(total_pixel_count)
    print("Collab Energy Consumption: {}".format(collab_energy_consumption))
    energy_savings = (100 * (initial_energy_consumption - collab_energy_consumption)) / initial_energy_consumption
    print("Energy Savings: {}".format(energy_savings))
