from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
import pickle

lower_bound = 82.
upper_bound = 220.

if __name__ == "__main__":

    brain_num = ["075", "100", "125", "150", "163", "175", "188", "200", "225",
                "250", "275", "300", "350", "425", "450"]
    "TRAIN"        
    for u in range(0, len(brain_num)):
        blue = Image.open("input/unlab/17015_thio_" + brain_num[u] + ".tif")
        bw_im = blue.convert('L')
        bw = np.array(bw_im)
        mean_ = np.mean(bw)
        for i in range(0, len(bw)):
            for j in range(0, len(bw[i])):
                val_gry = bw[i, j]
                if val_gry > upper_bound:
                    bw[i, j] = 0.
                elif val_gry < lower_bound:
                    bw[i, j] = 0.
        mask = bw
        bw[bw > 0.] = 255.
        bw = Image.fromarray(bw)
        bw.save("input/matrix/"+str(brain_num[u])+"mask.tif", "TIFF")
        bw.close()
        mask[mask > 0] = 1
        f = open('input/matrix/' + brain_num[u] + 'mask.pckl', 'wb')
        pickle.dump(mask, f)
        f.close()
        g = open('input/matrix/' + brain_num[u] + 'mean.pckl', 'wb')
        pickle.dump(mean_, g)
        g.close()
        print("(づ｡◕‿‿◕｡)づ " + str(brain_num[u]))



    brain_num = ["375", "400"]
    "TEST"        
    for o in range(0, len(brain_num)):
        blue = Image.open("input/brain/17015_thio_" + brain_num[o] + ".tif")
        bw_im = blue.convert('L')
        bw = np.array(bw_im)
        mean_ = np.mean(bw)

        for i in range(0, len(bw)):
            for j in range(0, len(bw[i])):
                val_gry = bw[i, j]
                if val_gry > upper_bound:
                    bw[i, j] = 0.
                elif val_gry < lower_bound:
                    bw[i, j] = 0.
        mask = bw
        bw[bw > 0.] = 255.
        bw = Image.fromarray(bw)
        bw.save("input/matrix/"+str(brain_num[o])+"mask.tif", "TIFF")
        bw.close()

        mask[mask > 0] = 1
        f = open('input/matrix/' +brain_num[o] + 'mean.pckl', 'wb')
        pickle.dump(mask, f)
        f.close()

        g = open('input/matrix/' + brain_num[o] + 'mean.pckl', 'wb')
        pickle.dump(mean_, g)
        g.close()

        print("(づ｡◕‿‿◕｡)づ " + str(brain_num[o]))
