import numpy as np
from PIL import Image

width = height = 512

# numpy (with for loops)
lol = np.zeros((width, height, 3), dtype=np.uint8)
for j in reversed(range(height)):
    print(f"Scanlines remaining: {j}")
    for i in range(width):
        r = i / width
        g = j / height
        b = .25
        lol[i, j] = (int(r * 255), int(g * 255), int(b * 255))
newImg1 = Image.fromarray(lol, mode="RGB")
newImg1.save("img1.png")
