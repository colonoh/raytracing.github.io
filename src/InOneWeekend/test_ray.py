import unittest

import taichi as ti
from taichi.math import vec3

from src.InOneWeekend.ray import Ray


class MyTestCase(unittest.TestCase):
    def test_ray(self):
        r = Ray(origifn=vec3(0.0, 0.0, 0.0), direction=vec3(1.0, 0.0, 0.0))

        @ti.kernel
        def test():
            result = r.at(1) #== vec3(1.0, 0.0, 0.0)
            assert result[0] == 1.0
            assert result[1] == 0.0
            assert result[2] == 0.0

        ti.init(arch=ti.cpu, debug=True)
        test()


if __name__ == '__main__':
    unittest.main()
