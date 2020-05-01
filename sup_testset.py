from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


"PARAMETERS"
pixel = 32
step_ = 32
dimension = pixel + pixel  # should be 20x20 = 400 
patch = dimension * dimension
perfect_patch = patch * 255
acceptance = perfect_patch * 98/100
brain_num = ["357"]
lower_bound = 82.
upper_bound = 220.
dim = pixel + pixel  
patch = dim * dim
zero = np.zeros(patch)
matrix = np.reshape(zero, (dim,dim))
black = Image.fromarray(matrix, mode='L')

try:
    os.mkdir("Testset/")
    os.mkdir("Testset/Test")
    os.mkdir("Testset/coordinates/")
    os.mkdir("Testset/Test/white")
    os.mkdir("Testset/Test/grey")
except OSError:
    print ("Creation of the directory Testset failed")


def create_matrix():

    brain_num = ["357"]
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
        f = open('input/matrix/' +brain_num[o] + 'mask.pckl', 'wb')
        pickle.dump(mask, f)
        f.close()

        g = open('input/matrix/' + brain_num[o] + 'mean.pckl', 'wb')
        pickle.dump(mean_, g)
        g.close()

        data = "(づ｡◕‿‿◕｡)づ " + str(brain_num[o])
        print(data)
        return data

def create_data():

    for i in range(0, len(brain_num)):
        img_white = 0
        img_grey = 0
        count = 0
        discarded = 0
        blue = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")#.convert('L')
        grey_mask = Image.open(r"input/mask/grey/17015_thio_" + brain_num[i] + "_grey_mask.tif").convert('L')
        white_mask = Image.open(r"input/mask/white/17015_thio_" + brain_num[i] + "_white_mask.tif").convert('L')
        mask = pickle.load(open("input/matrix/"+str(brain_num[i])+"mask.pckl", "rb"))
        mask[mask > 0] = 255.
        mask = Image.fromarray(mask, mode='L')

        white_result_array = np.zeros(1200)
        grey_result_array = np.zeros(1200)

        blue_np = np.array(blue)
        grey_mask_np = np.array(grey_mask)
        white_mask_np = np.array(white_mask)


        try:
            os.mkdir("Testset/brain/")
            os.mkdir("Testset/brain/white/")
            os.mkdir("Testset/brain/grey/")
        except OSError:
            print ("Creation of the directory brainnum failed")
        all_count = 0

        for x in range(pixel, blue.size[0] - pixel, step_):
            for y in range(pixel, blue.size[1] - pixel, step_):

                coord = ((x - pixel), (y - pixel), (x + pixel), (y + pixel))

                blue_box = blue.crop(coord)
                bw_box = blue_box.convert('L')
                white_mask_box = white_mask.crop(coord)
                grey_mask_box = grey_mask.crop(coord)
                blue_box_np = np.array(blue_box)   
                grey_mask_box_np = np.array(grey_mask_box)
                white_mask_box_np = np.array(white_mask_box)
                sum_mask_white = np.sum(white_mask_box_np)
                sum_mask_grey = np.sum(grey_mask_box_np)
                mask_box = mask.crop(coord)
                mask_box_np = np.array(mask_box)
                sum_mask = np.sum(mask_box_np)
                bw_matrix = np.array(bw_box)


                "WHITE MATTER ACCEPT"
                if sum_mask_white >= acceptance and sum_mask>= acceptance:
                    label = 0
                    img_white = img_white + 1
                    all_count += 1
                    blue_box.save("Testset/brain/white/" +str(all_count) + ".tif", "TIFF")
                    # blue_box.save("Testsetpatches/white/" + str(img_white) + "_" + brain_num[i] + ".tif", "TIFF")
                    with open("Testset/coordinates/the_all_file.txt", 'a') as the_all_file:
                        the_all_file.write(str(all_count) + '\t' + str(label)+'_'+ str(coord[0]) + '_' + str(coord[1])+ '_' + str(coord[2])+ '_' + str(coord[3]) +'\n')

                    with open("Testset/coordinates/"+brain_num[i]+'coord_white.txt', 'a') as the_file:
                        # the_file.write(str(coord[0]) + '\t' + str(coord[1])+ '\t' + str(coord[2])+ '\t' + str(coord[3])+'\t' +'\n')
                        the_file.write(str(img_white) + '\t' + str(coord[0]) + '\t' + str(coord[1])+ '\t' + str(coord[2])+ '\t' + str(coord[3])+'\n')         
                    "GREY MATTER ACCEPT"
                elif sum_mask_grey >= acceptance and sum_mask >= acceptance:
                    label = 1
                    img_grey = img_grey + 1
                    all_count += 1
                    blue_box.save("Testset/brain/grey/" +str(all_count)+ ".tif", "TIFF")
                    # blue_box.save("Testsetpatches/grey/" + str(img_grey)+"_"+brain_num[i]  + ".tif", "TIFF")
                    with open("Testset/coordinates/the_all_file.txt", 'a') as the_all_file:
                        the_all_file.write(str(all_count) + '\t' + str(label)+'\t'+ str(coord[0]) + '_' + str(coord[1])+ '_' + str(coord[2])+ '_' + str(coord[3])+'\n')

                    with open("Testset/coordinates/"+brain_num[i]+'coord_grey.txt', 'a') as the_file2:
                        # the_file2.write(str(coord[0]) + '\t' + str(coord[1])+ '\t' + str(coord[2])+ '\t' + str(coord[3])+'\t' +'\n')
                        the_file2.write(str(img_grey) + '\t' + str(coord[0]) + '\t' + str(coord[1])+ '\t' + str(coord[2])+ '\t' + str(coord[3])+'\n')

                else:
                    "NO"
                    discarded = discarded + 1
                count = count + 1
                blue_box.close()
                white_mask_box.close()
                grey_mask_box.close()
        the_all_file.close()
        the_file.close()
        the_file2.close()
        blue.close()
        grey_mask.close()
        white_mask.close()
        mask.close()

    data = "( ° ͜ʖ °) supervised done"
    return data

def coord():
    for i in range(0, len(brain_num)):
        blue = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")
        blue.save("Testset/" + brain_num[i]+"originalw.jpg")
        blue_clean = blue
        f = open("Testset/coordinates/"+brain_num[i]+"coord_white.txt", 'r')
        x = int(blue.size[0])
        y = int(blue.size[1])
        white = Image.new('RGB', (x, y), color = 'white')
        
        for line in f.readlines():
            x = line.split()
            # print(x)
            tup_ = (int(x[1]), int(x[2]), int(x[3]), int(x[4]))
            # print(tup_)
            box = blue_clean.crop(tup_)
            try:
                white.paste(box, tup_)
                blue.paste(black, tup_)
            except:
                print("fail but go on white")

        blue.save("Testset/" + brain_num[i]+"BLUEW.jpg")
        white.save("Testset/" + brain_num[i]+"WHITEW.jpg")
        blue_clean.close()
        white.close()
        blue.close()
        f.close()

    for i in range(0, len(brain_num)):
        blue = Image.open("input/brain/17015_thio_" + brain_num[i] + ".tif")
        blue.save("Testset/" + brain_num[i]+"originalg.jpg")
        blue_clean = blue

        g = open("Testset/coordinates/"+brain_num[i]+"coord_grey.txt", 'r')
        x = blue.size[0]
        y = blue.size[1]
        white = Image.new('RGB', (x, y), color = 'white')
        
        for line in g.readlines():
            x = line.split()
            img_ = int(x[0])
            tup_ = (int(x[1]), int(x[2]), int(x[3]), int(x[4]))
            box = blue_clean.crop(tup_)
            try:
                white.paste(box, tup_)
                blue.paste(black, tup_)
            except:
                print("fail but go on grey")

        blue.save("Testset/" + brain_num[i]+"BLUEG.jpg")
        white.save("Testset/" + brain_num[i]+"WHITEG.jpg")
        blue_clean.close()
        white.close()
        blue.close()
        g.close()
        data = "DONE"
        return data


if __name__ == "__main__":

    data = create_matrix()
    data = create_data()
    data = coord()
    print(data)

