from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

jpgfile = Image.open("./images/test_1.jpg").convert('L')
image = np.array(jpgfile.getdata()).reshape(jpgfile.size[0], jpgfile.size[1])
print(image.shape)
