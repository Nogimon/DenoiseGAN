#LD

import numpy as np
import cv2
from matplotlib import pyplot as plt
from Skimage import denoise_tv_chambolle

class DenoiseCv2:



    def __init__(self, directory):
        self.directory = directory

    def plotImage(self, img, name):
        plt.savefig("./" + name + ".png")



    def denoiseFastNIMeans(self):
        img = cv2.imread(self.directory)
        denoisedImg = cv2.fastNIMeansDenoisingColored(img, None, 10, 10, 7, 21)

        self.plotImage(img, "NIMeans")


class Skimage:

    def __init__(self, directory):
        self.directory = directory

    def denoiseTVC(self):
        img = 



if __name__ == "__main__":
    dir = "./try.png"
    denoiseCv2 = DenoiseCv2(dir)
    denoiseCv2.denoiseFastNIMeans()

