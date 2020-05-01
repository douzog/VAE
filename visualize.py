from PIL import Image, ImageDraw
import numpy as np
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pixel = 32
dim = pixel + pixel  
patch = dim * dim
zero = np.zeros(patch)
matrix = np.reshape(zero, (dim,dim))
black = Image.fromarray(matrix, mode='L')
pink = Image.new('RGB',(dim,dim), color = 'pink')
navy = Image.new('RGB',(dim,dim), color = 'navy')

# pink.show()
# navy.show()
# black = Image.open("boxthinnw.png")
# black.show()


import pandas as pd
data = pd.read_csv("cluster/cluster_results.txt", sep='\t')
data.columns = ["img", "true", "pred"]
# print(data)
data = data.sort_values(by=["img"])
# print(data)

try:
    os.mkdir("viz/")
except OSError:
    print ("viz results exist" )
    os.system("rm -r viz/")
    os.mkdir("viz/")

brain_num = ["357"]
for i in range(0, len(brain_num)):
    blue = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")
    blue.save("viz/"+brain_num[i]+"originalw.jpg")
    blue_clean = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")
    pred = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")
    f = open("Testset/coordinates/"+brain_num[i]+"coord_white.txt", 'r')
    x = int(blue.size[0])
    y = int(blue.size[1])
    white = Image.new('RGB', (x, y), color = 'white')
    
    for line in f.readlines():
        x = line.split()
        img = int(x[0])
        tup_ = (int(x[1]), int(x[2]), int(x[3]), int(x[4]))
        box = blue_clean.crop(tup_)

        loc = data.iloc[img]
        # print("loc", loc)
        # print("loc 0", loc[1])        
        # print("loc 1", loc[2])
        if loc[1] == loc[2]:
            pred.paste(pink,tup_)
        if loc[1] != loc[2]:   
            pred.paste(navy,tup_)
            white.paste(box, tup_)

        blue.paste(black, tup_)

    pred.save("viz/"+brain_num[i]+"PREDW.jpg")
    blue.save("viz/"+brain_num[i]+"BLUEW.jpg")
    white.save("viz/"+brain_num[i]+"WHITEW.jpg")
    blue_clean.close()
    white.close()
    pred.close()
    blue.close()
    f.close()

for i in range(0, len(brain_num)):
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

        loc = data.iloc[img]
        # print("loc", loc)
        # print("loc 0", loc[1])        
        # print("loc 1", loc[2])
        if loc[1] == loc[2]:
            pred.paste(pink,tup_)
        if loc[1] != loc[2]:
            pred.paste(navy,tup_)
            white.paste(box, tup_)

        blue.paste(black, tup_)



    blue.save("viz/"+ brain_num[i]+"BLUEG.jpg")
    white.save("viz/"+brain_num[i]+"WHITEG.jpg")
    pred.save("viz/"+brain_num[i]+"PREDG.jpg")
    pred.close()
    blue_clean.close()
    white.close()
    blue.close()
    g.close()


# file = open("cluster/cluster_results.txt")
# column = []
# for line in file:
#     # print(column)
#     column.append(int(line.split("\t")[0]))

# column.sort()
# print(column)

# file.close()