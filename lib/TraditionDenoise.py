#LD

import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.io
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
                                 #denoise_wavelet, estimate_sigma)

def plotImage(img, name):
    fig = plt.figure()
    plt.imshow(img)
    fig.savefig("./" + name + ".png")

class DenoiseCv2:



    def __init__(self, directory):
        self.directory = directory



    def denoiseFastNIMeans(self):
        img = cv2.imread(self.directory)
        denoisedImg = cv2.fastNlMeansDenoising(img, None, 10, 10, 7)

        plotImage(denoisedImg, "NIMeans")


class Skimage:

    def __init__(self, directory):
        self.directory = directory

    def denoiseTVC(self):
        img = skimage.io.imread(self.directory)
        plotImage(img, "tvc")




if __name__ == "__main__":
    dir = "./inputs.png"
    #denoiseCv2 = DenoiseCv2(dir)
    #denoiseCv2.denoiseFastNIMeans()
    denoiseSk = Skimage(dir)
    denoiseSk.denoiseTVC()


