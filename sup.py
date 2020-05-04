from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Author: Isabella Douzoglou

"PARAMETERS"
pixel = 32
step_ = 32
dimension = pixel + pixel  # should be 20x20 = 400 
patch = dimension * dimension
perfect_patch = patch * 255
acceptance = perfect_patch * 98/100
brain_num = ["375", "400"]

try:
    os.mkdir("Test/")
    os.mkdir("Testpatches/")
    os.mkdir("Testpatches/white/")
    os.mkdir("Testpatches/grey/")
    os.mkdir("Test/Test")
    os.mkdir("Test/coordinates/")
    os.mkdir("Test/Test/white")
    os.mkdir("Test/Test/grey")
except OSError:
    print ("Creation of the directory Test failed")


def create_data():

    for i in range(0, len(brain_num)):
        img_white = 0
        img_grey = 0
        count = 0
        discarded = 0
        # load image into pil object, load mask grey and white, as well as automated mask pickel matrix
        blue = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")#.convert('L')
        grey_mask = Image.open(r"input/mask/grey/17015_thio_" + brain_num[i] + "_grey_mask.tif").convert('L')
        white_mask = Image.open(r"input/mask/white/17015_thio_" + brain_num[i] + "_white_mask.tif").convert('L')
        mask = pickle.load(open("input/matrix/"+str(brain_num[i])+"mask.pckl", "rb"))
        mask[mask > 0] = 255. # convert to white
        mask = Image.fromarray(mask, mode='L') # convert to image

        try:
            os.mkdir("Test/"+brain_num[i]+"/")
            os.mkdir("Test/"+brain_num[i]+"/white/")
            os.mkdir("Test/"+brain_num[i]+"/grey/")
            os.mkdir("Test/"+brain_num[i]+"/grey_con/")            
            os.mkdir("Test/"+brain_num[i]+"/white_con/")
        except OSError:
            print ("Creation of the directory brainnum failed")

        all_count = 0 

        for x in range(pixel, blue.size[0] - pixel, step_):
            for y in range(pixel, blue.size[1] - pixel, step_):

                coord = ((x - pixel), (y - pixel), (x + pixel), (y + pixel))
                blue_box = blue.crop(coord)
                white_mask_box = white_mask.crop(coord)
                grey_mask_box = grey_mask.crop(coord)
                grey_mask_box_np = np.array(grey_mask_box)
                white_mask_box_np = np.array(white_mask_box)
                sum_mask_white = np.sum(white_mask_box_np)
                sum_mask_grey = np.sum(grey_mask_box_np)
                mask_box = mask.crop(coord)
                mask_box_np = np.array(mask_box)
                sum_mask = np.sum(mask_box_np)

                if sum_mask_white >= acceptance and sum_mask>= acceptance:
                    "WHITE MATTER"
                    all_count += 1
                    img_white = img_white + 1

                    # save accepted patch onto direcotry
                    blue_box.save("Test/"+brain_num[i]+"/white/" + str(img_white) + "_" + brain_num[i] + ".tif", "TIFF")
                    blue_box.save("Testpatches/white/" + str(img_white) + "_" + brain_num[i] + ".tif", "TIFF")

                    # save coordinates into a file
                    with open("Test/coordinates/"+brain_num[i]+'coord_white.txt', 'a') as the_file:
                        the_file.write(str(img_white) + '\t' + str(coord[0]) + '\t' + str(coord[1])+ '\t' + str(coord[2])+ '\t' + str(coord[3])+'\t' +'\n')         
                
                elif sum_mask_grey >= acceptance and sum_mask >= acceptance:
                    "GREY MATTER"
                    all_count += 1
                    img_grey = img_grey + 1

                    # save accepted patch onto directory
                    blue_box.save("Test/"+brain_num[i]+"/grey/" + str(img_grey)+"_"+brain_num[i]  + ".tif", "TIFF")
                    blue_box.save("Testpatches/grey/" + str(img_grey)+"_"+brain_num[i]  + ".tif", "TIFF")

                    # save coordinate file
                    with open("Test/coordinates/"+brain_num[i]+'coord_grey.txt', 'a') as the_file2:
                        the_file2.write(str(img_grey) + '\t' + str(coord[0]) + '\t' + str(coord[1])+ '\t' + str(coord[2])+ '\t' + str(coord[3])+'\t' +'\n')

                else:
                    discarded = discarded + 1
                count = count + 1
                blue_box.close()
                white_mask_box.close()
                grey_mask_box.close()
        
        the_file.close()
        the_file2.close()
        blue.close()
        grey_mask.close()
        white_mask.close()
        mask.close()

    data = "( ° ͜ʖ °) supervised done"
    return data

if __name__ == "__main__":

    data = create_data()
    print(data)

