import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import taichi as ti
from taichi.math import vec3

from src.InOneWeekend.ray import Ray


@ti.func
def ray_color(ray: Ray) -> vec3:
    t = 0.5 * (ray.direction.normalized().y + 1.0)
    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0)


def main():
    # Image
    aspect_ratio = 16.0 / 9.0
    image_width = 400
    image_height = int(image_width / aspect_ratio)

    # Camera
    viewport_height = 2.0
    viewport_width = aspect_ratio * viewport_height
    focal_length = 1.0

    origin = vec3(0., 0., 0.)
    horizontal = vec3(viewport_width, 0., 0.)
    vertical = vec3(0, viewport_height, 0.)
    lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length)

    start_taichi = time.time()
    ti.init()

    pixels = ti.Vector.field(n=3, dtype=ti.u8, shape=(image_height, image_width))  # note it's y, x

    @ti.kernel
    def render():
        for j, i in pixels:
            u = i / (image_width - 1)
            v = j / (image_height - 1)
            r = Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin)
            pixels[j, i].rgb = ray_color(r) * 255

    render()
    newImg1 = Image.fromarray(pixels.to_numpy())
    plt.imshow(newImg1)
    plt.show()

    end_taichi = time.time()
    print(f"{end_taichi - start_taichi:.2f}s")


if __name__ == "__main__":
    main()
