import numpy


def bezier(img):
    p = numpy.random.random(4)
    return numpy.power(1-img, 3)*p[0] + 3*numpy.power(1-img, 2)*img*p[1]\
        + 3*(1-img)*numpy.power(img, 2)*p[2] + numpy.power(img,3)*p[3]