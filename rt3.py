from PIL import Image
from functools import reduce
import numpy as np
import time
import numbers

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def cross(self, other):
        return vec3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)
    def __abs__(self):
        return self.dot(self)
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
rgb = vec3

(w, h) = (400, 300)         # Screen size
L = vec3(10, 10, 15)        # my light position
E = vec3(0, 0, 1)     # Eye position
FARAWAY = 1.0e39            # an implausibly huge distance


def rotate_light(rot_arr):
    global L
    new_L = np.dot(rot_arr, np.array([L.x, L.y, L.z]))
    L = vec3(new_L[0], new_L[1], new_L[2])


def raytrace(O, D, scene, bounce = 0):
    # O is the ray origin, D is the normalized ray direction
    # scene is a list of Sphere objects (see below)
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene, bounce)
            color += cc.place(hit)
    return color


class Sphere:
    def __init__(self, center, r, diffuse, mirror=0.25):
        self.c = center
        self.r = r
        self.diffuse = diffuse
        self.mirror = mirror

    def rotate(self, rot_arr):
        new_c = np.dot(rot_arr, np.array([self.c.x, self.c.y, self.c.z]))
        self.c = vec3(new_c[0], new_c[1], new_c[2])

    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                         # intersection point
        N = (M - self.c) * (1. / self.r)        # normal
        toL = (L - M).norm()                    # direction to light
        toO = (E - M).norm()                    # direction to ray origin
        nudged = M + N * .0001                  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker

class Plane:

    def __init__(self, n, p, diffuse, mirror=0.25):
        self.n = n
        self.p = p
        self.diffuse = diffuse
        self.mirror = mirror

    def rotate(self, rot_arr):
        pass

    def intersect(self, O: vec3, D):
        res = (self.n.dot(O - self.p)) / (self.n.dot(D))
        return np.where(res < 0, -res, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)                     # intersection point
        toL = (L - M).norm()                # direction to light
        toO = (E - M).norm()                # direction to ray origin
        nudged = M + self.n * .0001         # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(self.n.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - self.n * 2 * D.dot(self.n)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = self.n.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color


class CheckeredPlane(Plane):

    def diffusecolor(self, M):
        checker = (np.floor((M.x * 2.0)).astype(int) % 2) == (np.floor((M.z * 2.0)).astype(int) % 2)
        return self.diffuse * checker


class Triangle:

    def __init__(self, p1: vec3, p2: vec3, p3: vec3, diffuse, mirror=0.25):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.diffuse = diffuse
        self.mirror = mirror

    def rotate(self, rot_arr):
        new_p1 = np.dot(rot_arr, np.array([self.p1.x, self.p1.y, self.p1.z]))
        self.p1 = vec3(new_p1[0], new_p1[1], new_p1[2])
        new_p2 = np.dot(rot_arr, np.array([self.p2.x, self.p2.y, self.p2.z]))
        self.p2 = vec3(new_p2[0], new_p2[1], new_p2[2])
        new_p3 = np.dot(rot_arr, np.array([self.p3.x, self.p3.y, self.p3.z]))
        self.p3 = vec3(new_p3[0], new_p3[1], new_p3[2])

    def intersect(self, O, D):
        # aufgespannte ebene ermitteln
        epsilon = 0.00001
        e1 = self.p2 - self.p1
        e2 = self.p3 - self.p1
        hh = D.cross(e2)
        a = e1.dot(hh)

        mask = (np.abs(a) > epsilon)

        if any(mask):
            ff = np.where(mask, 1.0 / a, 0.0)

            ss = O - self.p1
            uu = ss.dot(hh) * ff

            mask_u = (uu >= 0.0) & (uu <= 1.0) & mask

            q = ss.cross(e1)
            v = D.dot(q) * ff

            mask_v = (v >= 0.0) & (uu + v <= 1.0) & mask_u

            t = e2.dot(q) * ff

            mask_t = (t > epsilon) & mask_v

            return np.where(mask_t, t, FARAWAY)

        return FARAWAY

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene, bounce):
        M = (O + D * d)  # intersection point
        N = (self.p3 - self.p1).cross(self.p2 - self.p1).norm()     # normal
        toL = (L - M).norm()                                        # direction to light
        toO = (E - M).norm()                                        # direction to ray origin
        nudged = M + N * .0001                                      # M nudged to avoid itself


        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # looking at the "back" of the triangle
        # works with fixed camera and light
        if N.z < 0:
            seelight = False


        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 2:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += raytrace(nudged, rayD, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 100) * seelight
        return color


def test_scene():
    scene = [
        Sphere(vec3(.75, .1, 1), .6, vec3(0, 0, 1)),
        Sphere(vec3(-.75, .1, 2.25), .6, vec3(.5, .223, .5)),
        Sphere(vec3(-2.75, .1, 3.5), .6, vec3(1, .572, .184)),
        CheckeredSphere(vec3(0,-9.5, 0), 9, vec3(.75, .75, .75), 0.25),
        ]

    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1.
    S = (-1, 1 / r + .25, 1, -1 / r + .25)
    x = np.tile(np.linspace(S[0], S[2], w), h)
    y = np.repeat(np.linspace(S[1], S[3], h), w)

    t0 = time.time()
    Q = vec3(x, y, 0)
    color = raytrace(E, (Q - E).norm(), scene)
    print ("Took", time.time() - t0)

    rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
    im = Image.merge("RGB", rgb).save("rt3.png")
    # im.show()
    return np.array(im)

