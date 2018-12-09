import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import os


if __name__ == "__main__":

    #mode = 'tiff'
    #mode = 'manytiff'
    mode = 'tiffinfolder'

    #used to process many tiff data under folders
    if (mode == 'tiffinfolder'):
        folderlist = os.listdir("./preprocesseddata/tiff3")
        for item in folderlist:
            filebase = "./preprocesseddata/tiff3/" + item
            filename = os.listdir(filebase)[0]
            file = filebase + "/" + filename
            print("the file being processed is:" + file)

            img = io.imread(file)
            low = img[2]
            imgave = np.mean(img[0:199,:,:], axis = 0)
            imgave = imgave.astype('uint8')

            io.imsave("./data/DENOISE_HR/temp/" + filename[:-5] + ".png", imgave)
            io.imsave("./data/DENOISE_LR/temp/" + filename[:-5] + ".png", low)

    #used to process many tiff data under folders
    #use to generate more than one from each tiff file
    if (mode == 'tiffinfoldermore'):
        folderlist = os.listdir("./preprocesseddata/tiff3")
        for item in folderlist:
            filebase = "./preprocesseddata/tiff3/" + item
            filename = os.listdir(filebase)[0]
            file = filebase + "/" + filename
            print("the file being processed is:" + file)

            img = io.imread(file)
            low = img[100]
            imgave = np.mean(img[0:200,:,:], axis = 0)
            imgave = imgave.astype('uint8')

            lows = []
            for i in range(5):
                low = img[0 + i*20]
                io.imsave("./data/DENOISE_HR_1001/" + filename[:-5] + str(i) + ".png", imgave)
                io.imsave("./data/DENOISE_LR_1001/" + filename[:-5] + str(i) + ".png", low)

            #io.imsave("./data/DENOISE_HR/temp/" + filename[:-5] + ".png", imgave)
            #io.imsave("./data/DENOISE_LR/temp/" + filename[:-5] + ".png", low)



    #aug mode do augmentation to denoise data
    #resize to generate another image and save
    if (mode == 'aug'):
        lowresdir = "./preprocesseddata/paper01.png"
        highresdir = "./preprocesseddata/paper01avg.png"
        low = io.imread(lowresdir)
        high = io.imread(highresdir)

        #high = high[::2, ::2]
        #low = low[::2, ::2]

        io.imsave("./data/DENOISE_HR/skinaug04.png", high)
        io.imsave("./data/DENOISE_LR/skinaug04.png", low)


    #old code
    #to generate multiple images from one tiff
    if (mode == 'manytiff'):
        name = 'Tape0524'
        tiffdir = './preprocesseddata/' + name + '.tiff'

        img = io.imread(tiffdir)
        
        imgave = np.mean(img[50:250,:,:], axis = 0)
        imgave = imgave.astype('uint8')
        plt.imshow(imgave)
        
        for i in range(50, 200):
            img1 = img[i,:,:]
            io.imsave("./data/DENOISE_HR_repeat/" + name + "{}.png".format(format(i, '03')), imgave)
            io.imsave("./data/DENOISE_LR_repeat/" + name + "{}.png".format(format(i, '03')), img1)
        



    #used for SR
    if (mode == 'tiff'):
        name = 'skin'
        lowresdir = "./preprocesseddata/" + name + "100.tiff"
        highresdir = "./preprocesseddata/" + name + "400.tiff"

        low = io.imread(lowresdir)
        high = io.imread(highresdir)

        for i in range(125, 571):
            highframe = high[:, i, :]
            lowframe = low[:, i, :]
            io.imsave("./data/OCMTRY_HR/" + name + "{}.png".format(i, '05'), highframe)
            io.imsave("./data/OCMTRY_LR/" + name + "{}.png".format(i, '05'), lowframe)

    #used for SR
    if (mode == 'png'):
        lowresdir = "./preprocesseddata/skin02.png"
        highresdir = "./preprocesseddata/skin02avg.png"
        low = io.imread(lowresdir)
        #io.imshow(low)
        high = io.imread(highresdir)
        start1 = 000
        end1 = 800
        start2 = 000
        end2 = 400
        high = high[start1:end1, start2:end2]
        low = low[start1:end1, start2:end2]

        low = low[::4, ::4]
        io.imshow(low)
        #io.imshow(high)
        io.imsave("./data/OCMTRY_HR/cross0011.png", high)
        io.imsave("./data/OCMTRY_LR/cross0011.png", low)





