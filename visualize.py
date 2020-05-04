from PIL import Image, ImageDraw
import numpy as np
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Author: Isabella Douzoglou

pixel = 32
dim = pixel + pixel  
patch = dim * dim
zero = np.zeros(patch)
matrix = np.reshape(zero, (dim,dim)) # make an empty matrix 
black = Image.fromarray(matrix, mode='L') # use matrix to make black patches
pink = Image.new('RGB',(dim,dim), color = 'pink') # create color patches to visualize
navy = Image.new('RGB',(dim,dim), color = 'navy')

import pandas as pd
data = pd.read_csv("cluster/cluster_results.txt", sep='\t') # use pandas to load the results of clustering 
data.columns = ["img", "true", "pred"]
data = data.sort_values(by=["img"]) # sort filenumbers to correspond to the coordinate all file

try:
    os.mkdir("viz/")
except OSError:
    print ("viz results exist" )
    os.system("rm -r viz/")
    os.mkdir("viz/")

brain_num = ["357"] # test brain number

for i in range(0, len(brain_num)):
    blue = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif") # open image
    blue.save("viz/"+brain_num[i]+"originalw.jpg") # save original
    blue_clean = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif") # open same image without alteration
    pred = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif") # viz pred
    f = open("Testset/coordinates/"+brain_num[i]+"coord_white.txt", 'r')
    x = int(blue.size[0])
    y = int(blue.size[1])
    white = Image.new('RGB', (x, y), color = 'white')
    
    for line in f.readlines():
        x = line.split()
        img = int(x[0])
        tup_ = (int(x[1]), int(x[2]), int(x[3]), int(x[4]))
        box = blue_clean.crop(tup_)

        loc = data.iloc[img] # check the image prediction
        if loc[1] == loc[2]:
            pred.paste(pink,tup_) # if prediction equals truth paste correct patch
        if loc[1] != loc[2]:   
            pred.paste(navy,tup_) # else incorrect patch
            white.paste(box, tup_) # visuealize patch obly in white file

        blue.paste(black, tup_)
    # save the predictions viz, and the black and incorrect preditions into a blank white file
    pred.save("viz/"+brain_num[i]+"PREDW.jpg")
    blue.save("viz/"+brain_num[i]+"BLUEW.jpg")
    white.save("viz/"+brain_num[i]+"WHITEW.jpg")

    blue_clean.close()
    white.close()
    pred.close()
    blue.close()
    f.close()

for i in range(0, len(brain_num)):
    # similar to the method above though for the grey class
    blue = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")
    blue.save("viz/"+brain_num[i]+"originalg.jpg")
    blue_clean = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")
    pred = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")

    g = open("Testset/coordinates/"+brain_num[i]+"coord_grey.txt", 'r')
    x = blue.size[0]
    y = blue.size[1]
    white = Image.new('RGB', (x, y), color = 'white')
    
    for line in g.readlines():
        x = line.split()
        img_ = int(x[0])
        tup_ = (int(x[1]), int(x[2]), int(x[3]), int(x[4]))
        box = blue_clean.crop(tup_)

        loc = data.iloc[img] # check the image prediction
        if loc[1] == loc[2]:
            pred.paste(pink,tup_) # if prediction equals truth paste correct patch
        if loc[1] != loc[2]:   
            pred.paste(navy,tup_) # else incorrect patch
            white.paste(box, tup_) # visuealize patch obly in white file

        blue.paste(black, tup_) # paste black for all

    blue.save("viz/"+ brain_num[i]+"BLUEG.jpg")
    white.save("viz/"+brain_num[i]+"WHITEG.jpg")
    pred.save("viz/"+brain_num[i]+"PREDG.jpg")
    pred.close()
    blue_clean.close()
    white.close()
    blue.close()
    g.close()
