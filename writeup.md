**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.jpg "Undistorted"
[image2]: ./output_images/combined_thresh.jpg "Binary Example"
[image3]: ./output_images/unwarp.jpg "Warp Example"
[image4]: ./output_images/histogram_polyfit.jpg "Fit Visual"
[image5]: ./output_images/final_output.jpg "Output"

---
Link to my project [code](https://github.com/chaitanyar56/CarND-Advanced-Lane-Lines/blob/master/advancedLaneLines_pipeline.ipynb)

### Camera Calibration

The code for this step is contained in the 3rd code cell of the IPython notebook.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

### Pipeline (single images)

#### 1. Distortion Correction

Using the coefficents from the camera calibration distortion correction is performed by using `cv2.undistort()` in the code cell 6. Here is the example for distortion corrected image.

![alt text][image1]

#### 2. Binary thresholded Image

Combination of color (H,S) and absolute sobel x thresholds are used to generate a binary image in the code cell 8. Here is the example on a test image.

![alt text][image2]

#### 3. Perspective transform

The code for my perspective transform includes a function called `unwarp_img()`, which appears in code cell 10 of Iynb notebook.  The `unwarp_img()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points and applies a perspective transform to get the birds eye view.  I chose the hardcode the source and destination points in the following manner:

```python
h,w = image.shape[:2]
# Define 4 source points (which takes shape of trapezoid)
src = np.float32([[170, h], [550, 460],[745, 460], [1200, h]])
# Define 4 destination points (which takes shape of rectangle)
dst = np.float32([[100, h], [100, 0], [1100, 0], [1100, h]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 170, 720      | 100, 720        |
| 550, 460      | 100, 0      |
| 1100, 460     | 1100, 0      |
| 1200, 720      | 1100, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 4. Detecting lane line pixels and finding a Polynomial fit
 Applying histogram on the warped image gives the location of left and right lines, that information is used to define a search window to detect the location of left and right lane pixels. Using the coordinates of the pixels a second order polynomial fit is used to draw a smooth curve. Cell 13 has the code and example image is shown in the figure 4.

![alt text][image4]

#### 5. Radius of curvature
Radius of curvature is calculation and distance from center calculation is done in the code cell 31 with the help of lecture notes. Distance from the center is calculated as difference between the center of lanelines and center of the image.

#### 6. Pipeline
The Pipeline is defined in the code cell 32 and it is tested with test image 6 in code cell 33. Result for the pipeline is shown in the figure 5

![alt text][image5]

---

### Pipeline (video)
Here's a [link to my video result](https://github.com/chaitanyar56/CarND-Advanced-Lane-Lines/blob/master/processed_project_video.mp4)

---

### Discussion
It was really difficult to choose the combination of thresholding techniques and  values to detect lines in bright lighting conditions. Pipeline is likely to fail if there is too much curving in the road which can be corrected by using higher order polynomial fits. It is also going to fail in the road conditions having more signs than lane lines  i.e. (hov lanes, arrow directions ) in challenge video.  Discarding features based on the direction of gradient may solve the problem.
