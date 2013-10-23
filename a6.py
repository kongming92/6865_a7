#assignment 6 starter code
#by Abe Davis
#
# Student Name: Charles Liu
# MIT Email: cliu2014@mit.edu

import numpy as np
import math
from scipy import linalg

### HELPERS ###
def imIter(im):
    for y in range(0,im.shape[0]):
        for x in range(0,im.shape[1]):
            yield (y,x)

def clipX(im, x):
    return min(im.shape[1] - 1, max(x, 0))

def clipY(im, y):
    return min(im.shape[0] - 1, max(y, 0))

def pix(im, y, x):
    return im[clipY(im, y), clipX(im, x)]

def interpolateLin(im, y, x):
    # same as from previous assignment
    x0 = int(math.floor(x))
    x1 = int(math.ceil(x))
    y0 = int(math.floor(y))
    y1 = int(math.ceil(y))

    if x0 == x1:
        temp0 = pix(im, y0, x0)
        temp1 = pix(im, y1, x0)
    else:
        # Interpolate between x's at y = y0
        temp0 = (x1 - x) * pix(im, y0, x0) + (x - x0) * pix(im, y0, x1)
        # Interpolate between x's at y = y1
        temp1 = (x1 - x) * pix(im, y1, x0) + (x - x0) * pix(im, y1, x1)

    if y0 == y1:
        return temp0
    # Otherwise, interpolate between y's
    return (y1 - y) * temp0 + (y - y0) * temp1

def applyHomography(source, out, H, bilinear=False):
    # takes the image source, warps it by the homography H, and adds it to the composite out.
    # If bilinear=True use bilinear interpolation, otherwise use NN.
    # Keep in mind that we are iterating through the output image, and the transformation
    # from output pixels to source pixels is the inverse of the one from source pixels to the output.
    # Does not return anything.
    Hinv = linalg.inv(H)
    for y, x in imIter(out):
        yp, xp, w = Hinv.dot(np.array([y, x, 1.0]))
        yp, xp = (yp / w, xp / w)
        if yp >= 0 and yp < source.shape[0] and xp >= 0 and xp < source.shape[1]:
            if bilinear:
                out[y, x] = interpolateLin(source, yp, xp)
            else:
                out[y, x] = source[int(round(yp)), int(round(xp))]

def addConstraint(systm, i, constr):
    # Adds the constraint constr to the system of equations ststm.
    # constr is simply listOfPairs[i] from the argument to computeHomography.
    # This function should fill in 2 rows of systm.
    # We want the solution to our system to give us the elements of a homography
    # that maps constr[0] to constr[1]. Does not return anything
    y, x = (constr[0][0], constr[0][1])
    yp, xp = (constr[1][0], constr[1][1])
    systm[2*i] = np.array([y, x, 1, 0, 0, 0, -y*yp, -x*yp, -yp])
    systm[2*i + 1] = np.array([0, 0, 0, y, x, 1, -y*xp, -x*xp, -xp])

def computeHomography(listOfPairs):
    # Computes and returns the homography that warps points listOfPairs[-][0] to listOfPairs[-][1]
    systm = np.zeros((9,9))
    for i, pair in enumerate(listOfPairs):
        addConstraint(systm, i, pair)
    systm[8,8] = 1.0
    if linalg.det(systm) == 0:
        return np.identity(3)
    RHS = np.array([0.0] * 8 + [1.0])
    return linalg.inv(systm).dot(RHS).reshape([3,3])

def computeTransformedBBox(imShape, H):
    # computes and returns [[ymin, xmin],[ymax,xmax]] for the transformed version of the rectangle described in imShape.
    # Keep in mind that when you usually compute H you want the homography that maps output pixels into source pixels,
    # whereas here we want to transform the corners of our source image into our output coordinate system.
    points = [[imShape[0], 0.0], [0.0, imShape[1]], [imShape[0], imShape[1]]]
    y, x, w = H.dot(np.array([0.0, 0.0, 1.0]))
    ymin = ymax = int(round(y / w))
    xmin = xmax = int(round(x / w))
    for point in points:
        newY, newX, newW = H.dot(np.array(point + [1.0]))
        newY, newX = (int(round(newY / newW)), int(round(newX / newW)))
        ymin, ymax, xmin, xmax = (min(ymin, newY), max(ymax, newY), min(xmin, newX), max(xmax, newX))
    return [[ymin, xmin], [ymax, xmax]]

def bboxUnion(B1, B2):
    # No, this is not a professional union for beat boxers. Though that would be awesome.
    # Rather, you should take two bounding boxes of the form [[ymin, xmin,],[ymax, xmax]]
    # and compute their union. Return a new bounding box of the same form. Beat boxing optional...
    return [[min(B1[0][0], B2[0][0]), min(B1[0][1], B2[0][1])],[max(B1[1][0], B2[1][0]), max(B1[1][1], B2[1][1])]]

def translate(bbox):
    # Takes a bounding box, returns a translation matrix that translates the top left corner of that bounding box to the origin.
    # This is a very short function.
    out = np.identity(3)
    out[0, 2] = -bbox[0][0]
    out[1, 2] = -bbox[0][1]
    return out

def stitch(im1, im2, listOfPairs):
    # Stitch im1 and im2 into a panorama. The resulting panorama should be in the coordinate system of im2,
    # though possibly extended to a larger image.
    # That is, im2 should never appear distorted in the resulting panorama, only possibly translated.
    # Returns the stitched output (which may be larger than either input image).
    H = computeHomography(listOfPairs)
    bbox = bboxUnion(computeTransformedBBox(im1.shape, H), [[0,0], [im2.shape[0], im2.shape[1]]])
    trans = translate(bbox)
    transInv = linalg.inv(trans)
    out = np.zeros((bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], 3))
    for y, x in imIter(out):
        yt, xt, wt = transInv.dot(np.array([y, x, 1.0]))
        if yt >= 0 and yt < im2.shape[0] and xt >= 0 and xt < im2.shape[1]:
            out[y, x] = interpolateLin(im2, yt, xt)
    applyHomography(im1, out, trans.dot(H) , True)
    return out

#######6.865 Only###############

def applyHomographyFast(source, out, H, bilinear=False):
    # takes the image source, warps it by the homography H, and adds it to the composite out.
    # This version should only iterate over the pixels inside the bounding box of source's image in out.
    bbox = computeTransformedBBox(source.shape, H)
    Hinv = linalg.inv(H)
    for y in xrange(bbox[0][0], bbox[1][0]):
        for x in xrange(bbox[0][1], bbox[1][1]):
            yp, xp, w = Hinv.dot(np.array([y, x, 1.0]))
            yp, xp = (yp / w, xp / w)
            if yp >= 0 and yp < source.shape[0] and xp >= 0 and xp < source.shape[1]:
                if bilinear:
                    out[y, x] = interpolateLin(source, yp, xp)
                else:
                    out[y, x] = source[int(round(yp)), int(round(xp))]


def computeNHomographies(listOfListOfPairs, refIndex):
    # This function takes a list of N-1 listOfPairs and an index.
    # It returns a list of N homographies corresponding to your N images.
    # The input N-1 listOfPairs describes all of the correspondences between images I(i) and I(i+1).
    # The index tells you which of the images should be used as a reference.
    # The homography returned for the reference image should be the identity.

    H_i = []
    N = len(listOfListOfPairs) + 1
    for i, listOfPairs in enumerate(listOfListOfPairs):
        H_i.append(computeHomography(listOfPairs))
    homographies =  [np.zeros((3,3))] * N
    homographies[refIndex] = np.identity(3)
    for i in xrange(N):
        if i < refIndex:
            homographies[i] = reduce(lambda x, y: x.dot(y), H_i[i:refIndex], np.identity(3))
        elif i > refIndex:
            if refIndex == 0:
                homographies[i] = reduce(lambda x, y: x.dot(linalg.inv(y)), H_i[i-1::-1], np.identity(3))
            else:
                homographies[i] = reduce(lambda x, y: x.dot(linalg.inv(y)), H_i[i-1:refIndex-1:-1], np.identity(3))
    return homographies

def compositeNImages(listOfImages, listOfH):
    # Computes the composite image. listOfH is of the form returned by computeNHomographies.
    # Hint: You will need to deal with bounding boxes and translations again in this function.
    bbox = computeTransformedBBox(listOfImages[0].shape, listOfH[0])
    for img, H in zip(listOfImages, listOfH):
        currentbbox = computeTransformedBBox(img.shape, H)
        bbox = bboxUnion(currentbbox, bbox)
    trans = translate(bbox)
    out = np.zeros((bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], 3))
    for img, H in zip(listOfImages, listOfH):
        applyHomographyFast(img, out, trans.dot(H), True)
    return out

def stitchN(listOfImages, listOfListOfPairs, refIndex):
    # Takes a list of N images, a list of N-1 listOfPairs, and the index of a reference image.
    # The listOfListOfPairs contains correspondences between each image Ii and image I(i+1).
    # The function should return a completed panorama
    homographies = computeNHomographies(listOfListOfPairs, refIndex)
    return compositeNImages(listOfImages, homographies)