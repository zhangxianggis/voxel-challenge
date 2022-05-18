from scene import Scene
import taichi as ti
from taichi.math import *
exposure = 1.0
petalParts = 5
leafParts = 4
scene = Scene(voxel_edges=0.08, exposure=exposure)
scene.set_floor(-0.7, (1.0, 1.0, 1.0))  # scene.set_floor(-0.7, (1, 0.8, 0.6))
scene.set_background_color((0.8, 0.8, 1.0))
scene.set_directional_light(
    (1.0, 1.5, 1.0), 0.2, vec3(1.0, 1.0, 1.0) / exposure)


@ti.func
def createPetal(pos, size, mat, color):
    for x, y in ti.ndrange((0, size), (0, size)):
        if ((x/size)**2+(y/size)**2)**2-(x/size)*(y/size) <= 0:
            d = x**2 + y**2
            z = 0.01 * d
            for offset_x, offset_y, offset_rotate in ti.ndrange((-1, 1), (-1, 1), (0, petalParts)):
                offset = vec3(offset_x, offset_y, 0)
                offset_rotate_ = 2 * pi * float(offset_rotate) / petalParts
                xyz = pos + offset + \
                    rotate3d(vec3(x, y, z), vec3(0, 0, 1), offset_rotate_)
                scene.set_voxel(xyz, mat, color)


@ti.func
def createLeave(pos, size, mat, color):
    for x, y in ti.ndrange((0, size), (0, size)):
        if (x/size)**3+(y/size)**3-(x/size)*(y/size) <= 0:
            d = (x-size*0.5)**2 + (y-size*0.5)**2
            z = -0.01 * d
            for offset_rotate in ti.ndrange((0, leafParts)):
                offset_rotate_ = 2 * pi * float(offset_rotate) / leafParts
                xyz = rotate3d(vec3(x, y, z), vec3(0, 0, 1), offset_rotate_)
                scene.set_voxel(vec3(xyz[0], xyz[2], xyz[1])+pos, mat, color)


@ti.func
def createSphere(pos, r, mat, color):
    for i, j, k in ti.ndrange((-64, 64), (-64, 64), (-64, 64)):
        if (i-pos[0])**2+(j-pos[1])**2+(k-pos[2])**2 <= r*r:
            scene.set_voxel(vec3(i, j, k), mat, color)


@ti.func
def createTrunk(pos, radius, height, mat, color):
    for x, z, y in ti.ndrange((-radius, radius), (-radius, radius), (-height, height)):
        if x**2 + z ** 2 <= radius ** 2:
            scene.set_voxel(pos + vec3(x, y, z), mat, color)


@ti.func
def createFloor():
    for i, j, k in ti.ndrange((-64, 64), (-64, 64), (-64, 64)):
        h = 3*ti.abs(ti.sin(i/16*pi)*ti.sin(j/16*pi))-42
        if k < h and k > -63:
            scene.set_voxel(vec3(i, k, j), 1, vec3(0.42, .62, 1.))
            if i % 2 == 0 and j % 2 == 0:
                scene.set_voxel(vec3(i, k, j), 1, vec3(1.0, 1.0, 1.0))


@ti.kernel
def initialize_voxels():
    # Your code here! :-)
    createTrunk(vec3(0, -18, 0), 2, 18, 1, vec3(0.5, 0.25, 0.0))
    createPetal(vec3(0, 24, 0), 48, 1, vec3(1.0, 0.0, 0.0))
    createSphere(vec3(0, 24, 0), 12, 1, vec3(1, 1, 0))
    createLeave(vec3(0, -20, 0), 64, 1, vec3(0.0, 1, 0.0))
    createFloor()


initialize_voxels()

scene.finish()
