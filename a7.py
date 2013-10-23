import numpy as np
from scipy import ndimage, linalg
import random
import a6
import math

class point():
  def __init__(self, x, y):
    self.x=x
    self.y=y

class feature():
  def __init__(self, pt, descriptor):
    self.pt=pt
    self.descriptor=descriptor

class correspondence():
  def __init__(self, pt1, pt2):
    self.pt1=pt1
    self.pt2=pt2

### HELPERS

def BW(im, weights=[0.3,0.6,0.1]):
  out = np.zeros((im.shape[0], im.shape[1]))
  (height, width, rgb) = np.shape(im)
  for y in xrange(height):
    for x in xrange(width):
      out[y][x] = np.dot(im[y][x], weights)
  return out

def imIter(im):
  for y in range(0,im.shape[0]):
      for x in range(0,im.shape[1]):
          yield (y,x)

### END HELPERS

def computeTensor(im, sigmaG=1, factorSigma=4):
  '''im_out: 3-channel-2D array. The three channels are Ixx, Ixy, Iyy'''
  lumiBlurred = ndimage.filters.gaussian_filter(BW(im), sigmaG)
  sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
  gradX = ndimage.filters.convolve(lumiBlurred, sobel, mode='reflect')
  gradY = ndimage.filters.convolve(lumiBlurred, np.transpose(sobel), mode='reflect')
  im_out = np.zeros(im.shape)
  for y, x in imIter(im_out):
    Ix = gradX[y, x]
    Iy = gradY[y, x]
    im_out[y, x] = np.array([Ix**2, Ix*Iy, Iy**2])
  s = sigmaG * factorSigma
  im_out = ndimage.filters.gaussian_filter(im_out, [s, s, 0])
  return im_out

def cornerResponse(im, k=0.15, sigmaG=1, factorSigma=4):
  '''resp: 2D array charactering the response'''
  tensor = computeTensor(im, sigmaG, factorSigma)
  resp  = np.zeros((im.shape[0], im.shape[1]))
  for y, x in imIter(resp):
    M = np.array([[tensor[y, x, 0], tensor[y, x, 1]], [tensor[y, x, 1], tensor[y, x, 2]]])
    R = linalg.det(M) - k * (np.trace(M)**2)
    if R > 0:
      resp[y, x] = R
  return resp

def HarrisCorners(im, k=0.15, sigmaG=1, factor=4, maxiDiam=7, boundarySize=5):
  '''result: a list of points that locate the images' corners'''
  possibleCorners = cornerResponse(im, k, sigmaG, factor)
  maxima = ndimage.filters.maximum_filter(possibleCorners, maxiDiam)
  corners = (possibleCorners == maxima)
  corners[possibleCorners == 0] = False
  height, width = corners.shape
  corners[0:boundarySize] = corners[height-1:height-boundarySize-1:-1] = False
  corners[:,0:boundarySize] = corners[:,width-1:width-boundarySize-1:-1] = False
  return [point(x[1], x[0]) for x in np.transpose(np.nonzero(corners))]

def computeFeatures(im, cornerL, sigmaBlurDescriptor=0.5, radiusDescriptor=4):
  '''f_list: a list of feature objects'''
  lumiBlurred = ndimage.filters.gaussian_filter(BW(im), sigmaBlurDescriptor)
  f_list = []
  for corner in cornerL:
    desc = descriptor(lumiBlurred, corner, radiusDescriptor)
    desc = 1.0/desc.std() * (desc - desc.mean())
    f_list.append(feature(corner, desc))
  return f_list

def descriptor(blurredIm, P, radiusDescriptor=4):
  '''patch: descriptor around 2-D point P, with size (2*radiusDescriptor+1)^2 in 1-D'''
  return blurredIm[P.y-radiusDescriptor:P.y+radiusDescriptor+1, P.x-radiusDescriptor:P.x+radiusDescriptor+1].flatten()

def findCorrespondences(listFeatures1, listFeatures2, threshold=1.7):
  '''correpondences: a list of correspondences object that associate two feature lists.'''
  features = [findCorrespondence(feature1, listFeatures2, threshold) for feature1 in listFeatures1]
  return filter(lambda x: x != None, features)

def findCorrespondence(feature1, listFeatures2, threshold=1.7):
  sumSquares = [sum((feature1.descriptor - feature2.descriptor)**2) for feature2 in listFeatures2]
  minIndex, minDistSq = (np.argmin(sumSquares), np.min(sumSquares))
  del sumSquares[minIndex]
  secondMinDistSq = np.min(sumSquares)
  if secondMinDistSq / minDistSq < threshold**2:
    return None
  return correspondence(feature1.pt, listFeatures2[minIndex].pt)

def RANSAC(listOfCorrespondences, Niter=5000, epsilon=4, acceptableProbFailure=1e-9):
  '''H_best: the best estimation of homorgraphy (3-by-3 matrix)'''
  '''inliers: A list of booleans that describe whether the element in listOfCorrespondences
  an inlier or not'''
  ''' 6.815 can bypass acceptableProbFailure'''
  numInliers = 0
  inliers = [False] * len(listOfCorrespondences)
  H_best = np.zeros((3,3))

  for n in xrange(Niter):
    correspondences = random.sample(listOfCorrespondences, 4)
    listOfPairs = A7PairsToA6Pairs(correspondences)
    H = a6.computeHomography(listOfPairs)
    currentInliers = []
    for c in listOfCorrespondences:
      estimate = H.dot(A7PointToA6Point(c.pt1))
      estimate[0], estimate[1] = (estimate[0] / estimate[2], estimate[1] / estimate[2])
      actual = A7PointToA6Point(c.pt2)
      error = math.sqrt(sum((estimate - actual)**2))
      currentInliers.append(bool(error < epsilon))
    currentNumInliers = sum(bool(x) for x in currentInliers)
    if currentNumInliers > numInliers:
      numInliers = currentNumInliers
      inliers = currentInliers
      H_best = H.copy()
    if (1 - (float(numInliers) / len(listOfCorrespondences))**4) ** n < acceptableProbFailure:
      break
  return (H_best, inliers)

def computeNHomographies(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  '''H_list: a list of Homorgraphy relative to L[refIndex]'''
  '''Note: len(H_list) is equal to len(L)'''
  H_i = []
  for i in xrange(len(L) - 1):
    im1Features = computeFeatures(L[i], HarrisCorners(L[i]), blurDescriptor, radiusDescriptor)
    im2Features = computeFeatures(L[i+1], HarrisCorners(L[i+1]), blurDescriptor, radiusDescriptor)
    correspondences = findCorrespondences(im1Features, im2Features)
    H_i.append(RANSAC(correspondences)[0])
  N = len(L)
  homographies = [np.zeros((3,3))] * N
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

def autostitch(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  '''Use your a6 code to stitch the images. You need to hand in your A6 code'''
  H_list = computeNHomographies(L, refIndex, blurDescriptor, radiusDescriptor)
  return a6.compositeNImages(L, H_list)

def weight_map(h,w):
  ''' Given the image dimension h and w, return the hxwx3 weight map for linear blending'''
  return w_map

def linear_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  ''' Return the stitching result with linear blending'''
  return out

def two_scale_blending(L, refIndex, blurDescriptor=0.5, radiusDescriptor=4):
  ''' Return the stitching result with two scale blending'''
  return out

# Helpers, you may use the following scripts for convenience.
def A7PointToA6Point(a7_point):
  return np.array([a7_point.y, a7_point.x, 1.0], dtype=np.float64)

def A7PairsToA6Pairs(a7_pairs):
  A7pointList1=map(lambda pair: pair.pt1 ,a7_pairs)
  A6pointList1=map(A7PointToA6Point, A7pointList1)
  A7pointList2=map(lambda pair: pair.pt2 ,a7_pairs)
  A6pointList2=map(A7PointToA6Point, A7pointList2)
  return zip(A6pointList1, A6pointList2)



