import numpy as np
from PIL import Image
import xarray as xr
import matplotlib.pyplot as plt


width = height = 512

# numpy (with for loops)
lol = np.zeros((width, height, 3), dtype=np.uint8)
for j in range(height):
    for i in range(width):
        r = i / width
        g = j / height
        b = .25
        lol[j, i] = (int(r * 255), int(g * 255), int(b * 255))

# newImg1 = Image.fromarray(lol)
# newImg1.save("img1.png")

da = xr.DataArray(
    data=lol,
    dims=["y", "x", "channel"]
    # coords=dict(
    #     red=["r", "g", "b"]
    # )
)
print(da)
da.plot.imshow()
plt.show()
