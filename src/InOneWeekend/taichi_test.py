import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import taichi as ti


width = height = 1024


# numpy (with for loops)
start_numpy = time.time()
# lol = np.zeros((width, height, 3), dtype=np.uint8)  # np.uint8 = 0..255
# for j in range(height):
#     for i in range(width):
#         r = i / width
#         g = j / height
#         b = .25
#         # faster than using .lo/etc.
#         lol[j, i] = (int(r * 255), int(g * 255), int(b * 255))  # r, g, b values
#
# newImg1 = Image.fromarray(lol)
# plt.imshow(newImg1)
# plt.show()

end_numpy = time.time()
print(f"numpy: {end_numpy - start_numpy}s")


###

start_taichi = time.time()
ti.init()

pixels = ti.Vector.field(n=3, dtype=ti.u8, shape=(width, height))

@ti.kernel
def paint():
    for i, j in pixels:
        r = i / width
        g = j / height
        b = .25
        pixels[j, i].rgb = r * 255, g * 255, b * 255

gui = ti.GUI("LOLCODE", res=(width, height))

paint()
# while gui.running:
#     gui.set_image(pixels)
#     gui.show()

newImg1 = Image.fromarray(pixels.to_numpy())
plt.imshow(newImg1)
plt.show()

end_taichi = time.time()
print(f"taichi: {end_taichi - start_taichi}s")
