from PIL import Image
import os
import numpy as np

img = Image.open(os.path.join("dataset/test/test100/benign/19.png"))

print(img.size)

img_array = np.array(img).transpose((2, 0, 1)) / 255

print(img_array.shape)
# print(img_array.max())

