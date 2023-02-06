import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import taichi as ti


start_taichi = time.time()

width = height = 512

ti.init()

pixels = ti.Vector.field(n=3, dtype=ti.u8, shape=(width, height))


@ti.kernel
def paint():
    for i, j in pixels:
        r = i / width
        g = j / height
        b = .25
        pixels[j, i].rgb = r * 255, g * 255, b * 255


paint()

newImg1 = Image.fromarray(pixels.to_numpy())
plt.imshow(newImg1)
plt.show()

end_taichi = time.time()
print(f"{end_taichi - start_taichi:.2f}s")
