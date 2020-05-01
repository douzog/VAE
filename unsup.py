import pickle
import os
import numpy as np
from PIL import Image
import pickle
from PIL import Image

"PICTURE SAMPLE PARAMETERS"
pixel = 32
step_ = 32
dimension = pixel + pixel  # should be 20x20 = 400 
patch = dimension * dimension
perfect_patch = patch * 255
acceptance = perfect_patch * 98/100
brain_number = ["075", "100", "125", "150", "163", "175", "188", "200", "225",
                "250", "275", "300", "350", "425", "450"]

try:
    os.mkdir("Train/")
    os.mkdir("Trainpatches/")
    os.mkdir("Trainpatches/noclass")
    os.mkdir("Train/Train/")
    os.mkdir("Train/coordinates/")
    os.mkdir("Train/Train_bw/")
    os.mkdir("Train/Train_new/")
    os.mkdir("Train/Train_blue/")

except OSError:
    print ("Creation of the directory Train failed" )

def create_data():
    mask_count = 0
    count = 0
    discarded = 0

    for i in range(0, len(brain_number)):
        blue = Image.open("input/unlab/17015_thio_" + brain_number[i] + ".tif")
        bw= blue.convert('L') 
        mask = pickle.load(open("input/matrix/"+str(brain_number[i])+"mask.pckl", "rb"))
        mask[mask > 0] = 255.
        mask = Image.fromarray(mask, mode='L')
        result_array = np.zeros(1200)

        try:
            os.mkdir("Train/Train_bw/"+brain_number[i]+"/")
            os.mkdir("Train/Train_blue/" +brain_number[i]+"/")
            os.mkdir("Train/Train_new/" +brain_number[i]+"/")

        except OSError:
            print ("Creation of the directory brain failed" )

        for x in range(pixel, blue.size[0] - pixel, step_):
            for y in range(pixel, blue.size[1] - pixel, step_):
                coord = ((x - pixel), (y - pixel), (x + pixel), (y + pixel))
                blue_box = blue.crop(coord)
                blue_box_np = np.array(blue_box)
                bw_box = bw.crop(coord)
                bw_matrix = np.array(bw_box)
                blue_np = np.array(blue_box)
                mask_box = mask.crop(coord)
                mask_box_np = np.array(mask_box)
                sum_mask = np.sum(mask_box_np)

                if sum_mask >= acceptance:
                    "YES"
                    mask_count = mask_count + 1
                    #  bw_box.save("Train/Train_bw/" + brain_number[i]+"/" + str(mask_count) + str(coord) + "_" + brain_number[i] + ".tif", "TIFF")
                    blue_box.save("Trainpatches/noclass/"  + str(mask_count) + ".tif", "TIFF")
                    with open("Train/coordinates/"+brain_number[i]+'coord.txt', 'a') as the_file:
                        the_file.write(str(mask_count) + '\t' + str(coord[0]) + '\t' + str(coord[1])+ '\t' + str(coord[2])+ '\t' + str(coord[3])+'\t' +'\n')
                else:
                    "NO"
                    discarded = discarded + 1
                count = count + 1
                blue_box.close()
                mask_box.close()

        blue.close()
        bw.close()
        mask.close()
        result_array = np.delete(result_array, 0, axis=0)
        np.save(brain_number[i]+".npy", result_array)
        data = "( -_-) " + str(brain_number[i])
        print(data)

    #save mask count
        with open("Train/"+ brain_number[i]+'total.txt', 'a') as the_file3:
            the_file3.write(str(mask_count)+'\n')
    data = "( ¬‿¬) unsupervised done "
    return data

def stack_data():
    input_train = np.load(brain_number[0]+".npy")
    for i in range(1, len(brain_number)):
        brain = np.load(brain_number[i]+".npy")
        input_train = np.vstack((input_train, brain))
    return input_train

if __name__ == "__main__":

    data = create_data()
    print(data)


