#LD

import numpy as np
import cv2
from matplotlib import pyplot as plt

class DenoiseCv2:

    def __init__(directory):
        self.directory = directory

    def plotImage(img, name):
        plt.savefig("./" + name + ".png")



    def denoiseFastNIMeans():
        img = cv2.read(self.directory)
        denoisedImg = cv2.fastNIMeansDenoisingColored(img, None, 10, 10, 7, 21)

        self.plotImage(img, "NIMeans")





if __name__ == "__main__":
    dir = ""
    denoiseCv2 = DenoiseCv2()
    denoiseCv2.denoiseFastNIMeans()

