with open("test.txt", 'w') as out:
    for i in range(1, 500):
        name = "../test_data_smu_setup/Overlap_Estimator/Sample_Images/viewer4/frame1_{}.jpg".format(i)
        out.write("{}\n".format(name))