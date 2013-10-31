def time_spent():
  '''N: # of hours you spent on this one'''
  return 16

def collaborators():
  '''Eg. ppl=['batman', 'ninja'] (use their athena username)'''
  return ['rlacey']

def potential_issues():
  return 'None that I am aware of, other than the occasional RANSAC weirdness'

def extra_credit():
#```` Return the function names you implemended````
#```` Eg. return ['full_sift', 'bundle_adjustment']````
  return []

def most_exciting():
  return 'My own panorama -- seeing a panorama of my house stitched together properly'

def most_difficult():
  return 'Spent a long time debugging the issue in RANSAC before Prof. Durand sent out the hint. In general, just putting everything together was challenging sometimes'

def my_panorama():
  input_images=['house1.png', 'house2.png', 'house3.png']
  output_images=['auto_stitch_house.png', 'linear_blending_house.png', 'two_scale_blending_house.png']
  return (input_images, output_images)

def my_debug():
  '''return (1) a string explaining how you debug
  (2) the images you used in debugging.

  Eg
  images=['debug1.jpg', 'debug2jpg']
  my_debug='I used blahblahblah...
  '''
  my_debug = '''
  I mostly used the Stata images to test corner detection and features. The images were nice because
  corners were clear and easy to find. It was easy to tell when a correspondence was incorrect.

  There weren't bugs that made me have to come up with my own images for the most part.

  I used the guedelon images to test for multiple stitching. I stitched them with a reasonable image as
  the reference (like image 1) but also with one of the end images as the reference, just to make sure
  that my functions were able to stitch when there were different numbers of images on each side.

  There were also times when I ran RANSAC, looked at the correspondences, and manually computed the
  homography matrix applied to various points to see if the warped points made sense. I also did the opposite;
  taking the homography from the points given in assignment 6, I computed the homography and compared
  it to the homography given by RANSAC. Taking into account randomness in RANSAC, I was fairly confident
  in the homography that was returned.
  '''

  images = ['stata-1.png', 'stata-2.png', 'guedelon-1.png', 'guedelon-2.png', 'guedelon-3.png', 'guedelon-4.png']
  return (my_debug, images)
