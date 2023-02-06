import numpy as np
from PIL import Image
import xarray as xr
import matplotlib.pyplot as plt


width = height = 512

# numpy (with for loops)
lol = np.zeros((width, height, 3), dtype=np.uint8)  # np.uint8 = 0..255
da = xr.DataArray(
    data=lol,
    dims=["y", "x", "channel"],
    coords=dict(
        channel=["red", "green", "blue"]
    )
)
for j in range(len(da.y)):  # really slow if you loop directly through da.y
    for i in range(len(da.x)):
        r = i / width
        g = j / height
        b = .25
        # faster than using .lo/etc.
        da.data[j, i] = (int(r * 255), int(g * 255), int(b * 255))  # r, g, b values

# newImg1 = Image.fromarray(lol)
# newImg1.save("img1.png")

print(da[0, 0].sel(channel="blue"))  # first pixel, only blue channel value
da.plot.imshow()
plt.show()
