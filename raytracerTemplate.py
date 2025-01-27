"""
/*******************************************************************************
 *
 *            #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *           %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *           %    %## #    CC        V    V  MM M  M MM RR    RR
 *            ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *            (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *              #%    %*    CCCCCC     VV    MM      MM RR    RR
 *             .%    %/
 *                (%.      Computer Vision & Mixed Reality Group
 *
 ******************************************************************************/
/**          @copyright:   Hochschule RheinMain,
 *                         University of Applied Sciences
 *              @author:   Prof. Dr. Ulrich Schwanecke, Fabian Stahl
 *             @version:   2.0
 *                @date:   01.04.2023
 ******************************************************************************/
/**         raytracerTemplate.py
 *
 *          Simple Python template to generate ray traced images and display
 *          results in a 2D scene using OpenGL.
 ****
"""
import rt3
from rendering import Scene, RenderWindow
from rt3 import *
import numpy as np


class RayTracer:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.scene = [CheckeredPlane(vec3(0, 1, 0), vec3(0, -1, 0), vec3(2, 2, 2)),
                      Sphere(vec3(0.3, 0, 0), .2, vec3(0, 0, 1)),
                      Sphere(vec3(-0.3, 0, 0), .2, vec3(1, 0, 0)),
                      Sphere(vec3(0, 0.4, 0), .2, vec3(0, 1, 0), 1),
                      Triangle(vec3(0.3, 0, 0), vec3(-0.3, 0, 0), vec3(0, 0.4, 0), vec3(1, 1, 0), 0)]
        self.center_of_gravity = vec3(0, 0, 0)
        self.e = rt3.E
        self.up = vec3(0, 1, 0)
        self.c = vec3(0, 0, 0)
        self.f = (self.c-self.e).norm()
        self.ratio = float(self.width) / self.height
        # Screen coordinates: x0, y0, x1, y1.
        self.screen = (-1, 1 / self.ratio + .25, 1, -1 / self.ratio + .25)
        self.theta = np.pi / 10

    def resize(self, new_width, new_height):
        self.width = new_width
        self.height = new_height

        self.ratio = float(self.width) / self.height
        self.screen = (-1, 1 / self.ratio + .25, 1, -1 / self.ratio + .25)

    def rotate_pos(self):

        rot_arr = np.array([
            [np.cos(self.theta), 0, np.sin(self.theta)],
            [0, 1, 0],
            [-np.sin(self.theta), 0, np.cos(self.theta)]
        ])
        self.rotate(rot_arr)

    def rotate_neg(self):

        rot_arr = np.array([
            [np.cos(-self.theta), 0, np.sin(-self.theta)],
            [0, 1, 0],
            [-np.sin(-self.theta), 0, np.cos(-self.theta)]
        ])
        self.rotate(rot_arr)

    def rotate(self, rot_arr):
        rt3.rotate_light(rot_arr)
        for o in self.scene:
            o.rotate(rot_arr)

    def render(self):

        all_x = np.tile(np.linspace(self.screen[0], self.screen[2], self.width), self.height)
        all_y = np.repeat(np.linspace(self.screen[1], self.screen[3], self.height), self.width)
        dir = vec3(all_x, all_y, 0)

        color = rt3.raytrace(self.e, (dir - self.e).norm(), self.scene)

        image = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((self.height, self.width))).astype(np.uint8), "L") for c in color.components()]

        return Image.merge("RGB", image)


# main function
if __name__ == '__main__':

    # set size of render viewport
    width, height = 640, 480

    # instantiate a ray tracer
    ray_tracer = RayTracer(width, height)

    # instantiate a scene
    scene = Scene(width, height, ray_tracer, "Raytracing Template")

    # pass the scene to a render window
    rw = RenderWindow(scene)

    # ... and start main loop
    rw.run()
