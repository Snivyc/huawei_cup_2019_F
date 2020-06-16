
import math
import numba
from numpy import sqrt, arcsin, cos, sin
# from math import asin as arcsin
import numpy as np

# def cal_length(a, b):
#     return np.linalg.norm(a - b)

import numba


@numba.jit
def cal_length(a, b):
    return np.sqrt((np.sum((b - a) ** 2)))


@numba.jit
def cal_add_delta(a, b):
    return 0.001 * cal_length(a, b)


def cal_length2(pa, pb, pc):

    r = 200
    rpb = pb
    xa, ya, za = pa
    xb, yb, zb = pb
    xc, yc, zc = pc

    a = (-za * (yb - yc) + zb * (ya - yc) - zc * (ya - yb)) / (
            xa * yb - xa * yc - xb * ya + xb * yc + xc * ya - xc * yb)
    b = (za * (xb - xc) - zb * (xa - xc) + zc * (xa - xb)) / (
                xa * yb - xa * yc - xb * ya + xb * yc + xc * ya - xc * yb)
    # print([[(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))/sqrt(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2), -sqrt(1 - 1/(b**2 + 1))*sqrt(-(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2/(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2) + 1), -sqrt(-(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2/(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2) + 1)/sqrt(b**2 + 1)], [0, 1/sqrt(b**2 + 1), -sqrt(1 - 1/(b**2 + 1))], [sqrt(-(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2/(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2) + 1), sqrt(1 - 1/(b**2 + 1))*(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))/sqrt(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2), (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))/(sqrt(b**2 + 1)*sqrt(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2))]])
    # if a < 0:
    f = np.array([a, b, 1], dtype=np.double)
    nf = np.linalg.norm(f)
    nf = f / nf

    nz = np.array([0, 0, 1], dtype=np.double)
    duichengzhou = (nf + nz) / np.linalg.norm(nf + nz)

    x, y, z = duichengzhou

    theta = np.pi

    xuanzhuan = np.array([[cos(theta) + (1 - cos(theta)) * (x ** 2), (1 - cos(theta)) * x * y - sin(theta) * z,
                           (1 - cos(theta)) * x * z + sin(theta) * y],
                          [(1 - cos(theta)) * y * x + sin(theta) * z, cos(theta) + (1 - cos(theta)) * (y ** 2),
                           (1 - cos(theta)) * y * z - sin(theta) * x],
                          [(1 - cos(theta)) * z * x - sin(theta) * y, (1 - cos(theta)) * z * y + sin(theta) * x,
                           cos(theta) + (1 - cos(theta)) * (z ** 2)]])

    fa = np.array([a, b, 1], dtype=np.double)



    # print(xuanzhuan.dot(fa))

    pb = pb.reshape((3, 1))
    pc = pc.reshape((3, 1))
    pa = pa.reshape((3, 1))

    pa = xuanzhuan.dot(pa)
    pb = xuanzhuan.dot(pb)
    pc = xuanzhuan.dot(pc)
    # print(pa, pb, pc)

    xa, ya, za = pa.reshape((3,))
    xb, yb, zb = pb.reshape((3,))
    xc, yc, zc = pc.reshape((3,))
    # print("ZZZZZZ")
    # print(za, zb, zc)
    # print(xa, ya, za)
    # print(xb, yb, zb)
    # print(xc, yc, zc)
    # print((yb - ya) / (xa - xb) > 0)

    if (yb - ya) * xc + (xa - xb) * yc + ya * xb - xa * yb > 0:
        t_flag = True
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # print("dayu0")
        ya, yb, yc = -ya, -yb, -yc
        if xa - xb < 0:
            t_flag = False
            ya, yb, yc = -ya, -yb, -yc



    else:
        t_flag = False
        if xa - xb < 0:
            t_flag = True
            ya, yb, yc = -ya, -yb, -yc
        # pass
        # print("xiaoyu0")
        # ya, yb,yc = -ya, -yb,-yc
    # print(xa, ya, za)
    # print(xb, yb, zb)
    # print(xc, yc, zc)

    xo = ((xa - xb) * (xa * xb - xb ** 2 + ya * yb - yb ** 2) + (ya - yb) * (r * xa * sqrt(
        (xa ** 2 - 2 * xa * xb + xb ** 2 + ya ** 2 - 2 * ya * yb + yb ** 2) / (
                xa ** 2 - 2 * xa * xb + xb ** 2)) - r * xb * sqrt(
        (xa ** 2 - 2 * xa * xb + xb ** 2 + ya ** 2 - 2 * ya * yb + yb ** 2) / (
                xa ** 2 - 2 * xa * xb + xb ** 2)) - xa * yb + xb * ya)) / ((xa - xb) ** 2 + (ya - yb) ** 2)

    yo = (-(xa - xb) * (r * xa * sqrt((xa ** 2 - 2 * xa * xb + xb ** 2 + ya ** 2 - 2 * ya * yb + yb ** 2) / (
            xa ** 2 - 2 * xa * xb + xb ** 2)) - r * xb * sqrt(
        (xa ** 2 - 2 * xa * xb + xb ** 2 + ya ** 2 - 2 * ya * yb + yb ** 2) / (
                xa ** 2 - 2 * xa * xb + xb ** 2)) - xa * yb + xb * ya) + (ya - yb) * (
                  xa * xb - xb ** 2 + ya * yb - yb ** 2)) / ((xa - xb) ** 2 + (ya - yb) ** 2)
    # print("o", xo, yo)
    xd1, yd1 = (r ** 2 * xc - r ** 2 * xo + r * yc * sqrt(
        -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) - r * yo * sqrt(
        -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + xc ** 2 * xo - 2 * xc * xo ** 2 + xo ** 3 + xo * yc ** 2 - 2 * xo * yc * yo + xo * yo ** 2) / (
                       xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2), (
                       r ** 2 * yc - r ** 2 * yo - r * (xc - xo) * sqrt(
                   -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + xc ** 2 * yo - 2 * xc * xo * yo + xo ** 2 * yo + yc ** 2 * yo - 2 * yc * yo ** 2 + yo ** 3) / (
                       xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2)

    xd2, yd2 = (r ** 2 * xc - r ** 2 * xo - r * yc * sqrt(
        -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + r * yo * sqrt(
        -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + xc ** 2 * xo - 2 * xc * xo ** 2 + xo ** 3 + xo * yc ** 2 - 2 * xo * yc * yo + xo * yo ** 2) / (
                       xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2), (
                       r ** 2 * yc - r ** 2 * yo + r * (xc - xo) * sqrt(
                   -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + xc ** 2 * yo - 2 * xc * xo * yo + xo ** 2 * yo + yc ** 2 * yo - 2 * yc * yo ** 2 + yo ** 3) / (
                       xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2)
    # print("d1", xd1, yd1)
    # print(xo, yo)
    # print("d2", xd2, yd2)
    t1 = ((xb - xa) * (xd1 - xb) + (yb - ya) * (yd1 - yb)) / sqrt(
        ((xb - xa) ** 2 + (yb - ya) ** 2) * ((xd1 - xb) ** 2 + (yd1 - yb) ** 2))
    t2 = ((xb - xa) * (xd2 - xb) + (yb - ya) * (yd2 - yb)) / sqrt(
        ((xb - xa) ** 2 + (yb - ya) ** 2) * ((xd2 - xb) ** 2 + (yd2 - yb) ** 2))
    # print(t1, t2)

    if t1 > t2:
        xd, yd = xd1, yd1
    else:
        xd, yd = xd2, yd2


    # po = np.array([xo, yo, za]).reshape(3, )
    # pd = np.array([xd, yd, za]).reshape(3, )

    # print("po", po)
    # print("pd", pd)
    # print(pb, rpb)
    # print(cal_length(po, pd))
    # print(cal_length(po, rpb))

    # assert xd == xd2
    # assert yd == yd2
    # print(sqrt((xb - xd) ** 2 + (yb - yd) ** 2))
    # print("d", xd, yd)
    # print(xd, yd)
    # print

    # print(xd1,yd1)
    # print(xd2,yd2)
    # print(xd,yd)
    return 2 * r * arcsin(sqrt((xb - xd) ** 2 + (yb - yd) ** 2) / (2 * r)) + sqrt((xd - xc) ** 2 + (yd - yc) ** 2)



def cal_add_delta2(a, b, c):
    return 0.001 * cal_length2(a, b, c)


def get_info(pa, pb, pc):
    r = 1000
    rpb = pb
    xa, ya, za = pa
    xb, yb, zb = pb
    xc, yc, zc = pc

    a = (-za * (yb - yc) + zb * (ya - yc) - zc * (ya - yb)) / (
            xa * yb - xa * yc - xb * ya + xb * yc + xc * ya - xc * yb)
    b = (za * (xb - xc) - zb * (xa - xc) + zc * (xa - xb)) / (xa * yb - xa * yc - xb * ya + xb * yc + xc * ya - xc * yb)
    # print([[(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))/sqrt(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2), -sqrt(1 - 1/(b**2 + 1))*sqrt(-(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2/(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2) + 1), -sqrt(-(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2/(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2) + 1)/sqrt(b**2 + 1)], [0, 1/sqrt(b**2 + 1), -sqrt(1 - 1/(b**2 + 1))], [sqrt(-(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2/(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2) + 1), sqrt(1 - 1/(b**2 + 1))*(b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))/sqrt(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2), (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))/(sqrt(b**2 + 1)*sqrt(a**2 + (b*sqrt(1 - 1/(b**2 + 1)) + 1/sqrt(b**2 + 1))**2 + (b/sqrt(b**2 + 1) - sqrt(1 - 1/(b**2 + 1)))**2))]])
    # if a < 0:
    f = np.array([a, b, 1], dtype=np.double)
    nf = np.linalg.norm(f)
    nf = f / nf

    nz = np.array([0,0,1], dtype = np.double)
    duichengzhou = (nf + nz) / np.linalg.norm(nf + nz)

    x,y,z = duichengzhou

    theta = np.pi


    xuanzhuan = np.array([[cos(theta) + (1 - cos(theta)) * (x ** 2), (1 - cos(theta)) * x * y - sin(theta) * z, (1 - cos(theta)) * x * z + sin(theta) * y],
                          [(1-cos(theta))*y*x+sin(theta)*z, cos(theta)+(1-cos(theta))*(y**2), (1-cos(theta)) * y*z - sin(theta) * x],
                          [(1-cos(theta))*z*x-sin(theta)*y, (1-cos(theta ))*z*y + sin(theta)*x, cos(theta)+(1-cos(theta))*(z**2)]])



    fa = np.array([a, b, 1], dtype=np.double)

    fa.resize((3, 1))


    # print(xuanzhuan.dot(fa))

    pa.resize((3, 1))
    pb.resize((3, 1))
    pc.resize((3, 1))

    pa = xuanzhuan.dot(pa)
    pb = xuanzhuan.dot(pb)
    pc = xuanzhuan.dot(pc)
    # print(pa, pb, pc)

    xa, ya, za = pa.reshape((3,))
    xb, yb, zb = pb.reshape((3,))
    xc, yc, zc = pc.reshape((3,))
    # print("ZZZZZZ")
    # print(za, zb, zc)
    # print(xa, ya, za)
    # print(xb, yb, zb)
    # print(xc, yc, zc)
    # print((yb - ya) / (xa - xb) > 0)

    if (yb - ya) * xc + (xa - xb) * yc + ya * xb - xa * yb > 0:
        t_flag = True
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # print("dayu0")
        ya, yb, yc = -ya, -yb, -yc
        if xa - xb < 0:
            t_flag = False
            ya, yb, yc = -ya, -yb, -yc



    else:
        t_flag = False
        if xa - xb < 0:
            t_flag = True
            ya, yb, yc = -ya, -yb, -yc
        # pass
        # print("xiaoyu0")
        # ya, yb,yc = -ya, -yb,-yc
    # print(xa, ya, za)
    # print(xb, yb, zb)
    # print(xc, yc, zc)

    xo = ((xa - xb) * (xa * xb - xb ** 2 + ya * yb - yb ** 2) + (ya - yb) * (r * xa * sqrt(
        (xa ** 2 - 2 * xa * xb + xb ** 2 + ya ** 2 - 2 * ya * yb + yb ** 2) / (
                xa ** 2 - 2 * xa * xb + xb ** 2)) - r * xb * sqrt(
        (xa ** 2 - 2 * xa * xb + xb ** 2 + ya ** 2 - 2 * ya * yb + yb ** 2) / (
                xa ** 2 - 2 * xa * xb + xb ** 2)) - xa * yb + xb * ya)) / ((xa - xb) ** 2 + (ya - yb) ** 2)

    yo = (-(xa - xb) * (r * xa * sqrt((xa ** 2 - 2 * xa * xb + xb ** 2 + ya ** 2 - 2 * ya * yb + yb ** 2) / (
            xa ** 2 - 2 * xa * xb + xb ** 2)) - r * xb * sqrt(
        (xa ** 2 - 2 * xa * xb + xb ** 2 + ya ** 2 - 2 * ya * yb + yb ** 2) / (
                xa ** 2 - 2 * xa * xb + xb ** 2)) - xa * yb + xb * ya) + (ya - yb) * (
                  xa * xb - xb ** 2 + ya * yb - yb ** 2)) / ((xa - xb) ** 2 + (ya - yb) ** 2)
    # print("o", xo, yo)
    xd1, yd1 = (r ** 2 * xc - r ** 2 * xo + r * yc * sqrt(
        -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) - r * yo * sqrt(
        -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + xc ** 2 * xo - 2 * xc * xo ** 2 + xo ** 3 + xo * yc ** 2 - 2 * xo * yc * yo + xo * yo ** 2) / (
                       xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2), (
                       r ** 2 * yc - r ** 2 * yo - r * (xc - xo) * sqrt(
                   -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + xc ** 2 * yo - 2 * xc * xo * yo + xo ** 2 * yo + yc ** 2 * yo - 2 * yc * yo ** 2 + yo ** 3) / (
                       xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2)

    xd2, yd2 = (r ** 2 * xc - r ** 2 * xo - r * yc * sqrt(
        -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + r * yo * sqrt(
        -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + xc ** 2 * xo - 2 * xc * xo ** 2 + xo ** 3 + xo * yc ** 2 - 2 * xo * yc * yo + xo * yo ** 2) / (
                       xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2), (
                       r ** 2 * yc - r ** 2 * yo + r * (xc - xo) * sqrt(
                   -r ** 2 + xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2) + xc ** 2 * yo - 2 * xc * xo * yo + xo ** 2 * yo + yc ** 2 * yo - 2 * yc * yo ** 2 + yo ** 3) / (
                       xc ** 2 - 2 * xc * xo + xo ** 2 + yc ** 2 - 2 * yc * yo + yo ** 2)
    # print("d1", xd1, yd1)
    # print(xo, yo)
    # print("d2", xd2, yd2)
    t1 = ((xb - xa) * (xd1 - xb) + (yb - ya) * (yd1 - yb)) / sqrt(
        ((xb - xa) ** 2 + (yb - ya) ** 2) * ((xd1 - xb) ** 2 + (yd1 - yb) ** 2))
    t2 = ((xb - xa) * (xd2 - xb) + (yb - ya) * (yd2 - yb)) / sqrt(
        ((xb - xa) ** 2 + (yb - ya) ** 2) * ((xd2 - xb) ** 2 + (yd2 - yb) ** 2))
    # print(t1, t2)

    if t1 > t2:
        xd, yd = xd1, yd1
    else:
        xd, yd = xd2, yd2

    if t_flag:
        ya, yb, yc, yd, yo = -ya, -yb, -yc, -yd, -yo

    xuanzhuan = np.linalg.inv(xuanzhuan)

    po = xuanzhuan.dot(np.array((xo, yo, za)).reshape(3, 1)).reshape(3, )

    pd = xuanzhuan.dot(np.array((xd, yd, za)).reshape(3, 1)).reshape(3, )

    pb = xuanzhuan.dot(np.array((xb, yb, zb)).reshape(3, 1)).reshape(3, )
    # po = np.array([xo, yo, za]).reshape(3, )
    # pd = np.array([xd, yd, za]).reshape(3, )

    # print("po", po)
    # print("pd", pd)
    # print(pb, rpb)
    # print(cal_length(po, pd))
    # print(cal_length(po, rpb))

    thetas = np.linspace(0, 2 * np.pi, 100)

    bp = []
    for theta in thetas:
        bp.append(
            xuanzhuan.dot(np.array((r * np.sin(theta) + xo, r * np.cos(theta) + yo, za)).reshape(3, 1)).reshape(3, ))

    x, y, z = [], [], []

    for i in bp:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    # assert xd == xd2
    # assert yd == yd2
    # print(sqrt((xb - xd) ** 2 + (yb - yd) ** 2))
    # print("d", xd, yd)
    # print(xd, yd)
    # print

    # print(xd1,yd1)
    # print(xd2,yd2)
    # print(xd,yd)
    return po, pd, (x, y, z)


