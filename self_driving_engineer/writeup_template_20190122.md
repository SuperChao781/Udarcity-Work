# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps:
1, I converted the images to grayscale，it is like this：

![](1_gray_figure.png)

2, I smooth the image by the function "cv2.GaussianBlur"，it is like this：

![](2_blur_figure.png)

3, I use the opencv canny function for edge detection on the smoothed image，it is like this：

![](3_canny_edge.png)


4，I make a regional selection of the results of the edge detecting.The region is a triangle,Take the bottom of the image as the bottom edge，Take 50 pixels downto the center of the image as the vertex.Only the test results in this area are retained.

The triangle region is like this(white region):

![](4_mask_area.png)

The masked region is like this:

![](4_canny_edge_mask.png)

5，I perform the Hough line transformation on the detection result after the region selection，it is like this:

![](5_hough_line.png)

6，Draw the Hough line transformation result on the original image，it is like this:

![](6_result.png)

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by following 6 steps:


1、First, summarize the endpoint information of each line segment obtained by the Hough_line transformation.

2、Calculate the slope (the value 'k'), intercept (the value 'b') and length (the value 'len') of each line segment；

3、Cluster all line segments based on (k,b) information，the rule of cluster is like this：

  （1）when the difference of k between the two line segments is <0.3 and the difference of b is <100, the two line segments are belong to the same class；

  （2）The line segments satisfying the condition are combined into one line segment group, and k, b of the line segment are counted, then we get the k-value boundary (kmin, kmax) and b-value boundary (bmin, bmax) of the line segment group；

  （3）After then, other line segments are directly judged with the line segment group. In following case we judge that the line segments is belong to the group: the 'k' value of the line segment is between (kmin-0.3,kmax+0.3)，the 'b' value of the line segment is between (bmin-100,bmax+100).If the two conditions is both satisfied, it is determined that this line segment belongs to the line segment group.In this case,we will update the k-value boundary and b-value boundary of the line segment group with the k and b values of the line segment.
  
  （4）In the above process, the line segment where k=0 or k does not exist is excluded;
 
   The lane line segment groups after the clustering of a certain frame picture is as follows（Each set of lane lines is represented by a different color）:

![](7_segments_groups.png)

4、After completing the clustering, for each segment group，Extract the x and y coordinates of all endpoints of the line segment within the group，do least squares fitting by the numpy.polyfit function，get the k, b values of the fitted line，then calculate the length of the line（The sum of the lengths of all the segments in the group）；

5、Find the endpoint for the above line:

One end:upper boundary of Y-axis,the value 'y1', set as "The apex of the area of masked region"（see pipeline step 3）；

The other end:Lower boundary of Y-axis,the value 'y2', set as the bottom of the image；

Calculate the corresponding abscissa x1 and x2 according to the straight line k-b, the formula is x=y/k-b;

The fitted line is as follows：

![](8_lines_before_reject.png)


6、Next, we perform validity judgment on the lines.We define 2 invalid  conditions:all lines with length <50 is removed；If the difference between k of any two straight lines is <0.3 and the b value difference is <100, the shorter one of the two straight lines is removed.
  It is like this:

![](9_lines_after_reject.png)


### 2. Identify potential shortcomings with your current pipeline

First shortcoming：All parameters are manually adjusted according to the input image, including：Canny detection threshold；the selection of region；threshold of Hough line transformation 、threshold of clustering,and so on.These parameters need to be re-adjusted if the image scene changes. Therefore, the generalization performance of this scheme is not good.

Second shortcoming：This scheme can only identify straight lane lines and cannot handle the curve information well.


### 3. Suggest possible improvements to your pipeline


An improved idea is to be able to identify the edge shape detected by Canny in a more intelligent way, identify the corresponding entities, such as vehicles, roadsides, sky-to-ground boundaries, and of course, lane lines; In the process, you should make full use of the visual characteristics of the lane lines, such as they can form a set of smooth curves in space (even if they are not continuous), they can only be located in the road surface, and so on.



