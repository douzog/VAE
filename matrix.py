from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
import pickle

# Author: Isabella Douzoglou

"THRESHOLD PARAMETERS"
lower_bound = 82.
upper_bound = 220.

if __name__ == "__main__":

    # brain specimen number list
    brain_num = ["075", "100", "125", "150", "163", "175", "188", "200", "225",
                "250", "275", "300", "350", "425", "450"]
    "TRAIN"        
    for u in range(0, len(brain_num)):
        blue = Image.open("input/unlab/17015_thio_" + brain_num[u] + ".tif") # open image
        bw_im = blue.convert('L') # convert to black
        bw = np.array(bw_im)
        mean_ = np.mean(bw)

        # for the image interity check if each pixel falls above or below the threshold set to zero
        for i in range(0, len(bw)):
            for j in range(0, len(bw[i])):
                val_gry = bw[i, j]
                if val_gry > upper_bound:
                    bw[i, j] = 0.
                elif val_gry < lower_bound:
                    bw[i, j] = 0.
                    
        mask = bw # make a mask
        bw[bw > 0.] = 255. # all items above zero to white
        bw = Image.fromarray(bw) # create image
        bw.save("input/matrix/"+str(brain_num[u])+"mask.tif", "TIFF") # save image mask
        bw.close()
        mask[mask > 0] = 1
        f = open('input/matrix/' + brain_num[u] + 'mask.pckl', 'wb') # save binary mask matrix
        pickle.dump(mask, f)
        f.close()
        g = open('input/matrix/' + brain_num[u] + 'mean.pckl', 'wb')
        pickle.dump(mean_, g)
        g.close()
        print("(づ｡◕‿‿◕｡)づ " + str(brain_num[u]))

    "TEST"        
    brain_num = ["375", "400"]
    for o in range(0, len(brain_num)):
        blue = Image.open("input/brain/17015_thio_" + brain_num[o] + ".tif")
        bw_im = blue.convert('L')
        bw = np.array(bw_im)
        mean_ = np.mean(bw)

        # for all pixels check if they fall above or below threshold parameters
        for i in range(0, len(bw)):
            for j in range(0, len(bw[i])):
                val_gry = bw[i, j]
                if val_gry > upper_bound:
                    bw[i, j] = 0.
                elif val_gry < lower_bound:
                    bw[i, j] = 0.

        mask = bw
        bw[bw > 0.] = 255. # convert all above 0 to white
        bw = Image.fromarray(bw)
        bw.save("input/matrix/"+str(brain_num[o])+"mask.tif", "TIFF") # save image mask
        bw.close()

        mask[mask > 0] = 1
        f = open('input/matrix/' +brain_num[o] + 'mean.pckl', 'wb') # save binary matrix
        pickle.dump(mask, f)
        f.close()

        g = open('input/matrix/' + brain_num[o] + 'mean.pckl', 'wb')
        pickle.dump(mean_, g)
        g.close()

        print("(づ｡◕‿‿◕｡)づ " + str(brain_num[o]))
