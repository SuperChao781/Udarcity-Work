
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')


# ## Read in an Image

# In[2]:

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
plt.title('original figure;')
plt.show()


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[18]:

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    segment_param_list = [];#线段参数列表 参数如下：xmin,ymin,xmax,ymax,k,b,length
    for line in lines:
        for x1,y1,x2,y2 in line:
            if(x1==x2):
                continue;
            elif(y1==y2):
                continue;
            else:
                k = (y2-y1)/(x2-x1);
                b = y2-(y2-y1)/(x2-x1)*x2;
                length = ((x2-x1)**2 + (y2-y1)**2 )**0.5;
                segment_param_list.append((x1,y1,x2,y2,k,b,length));

    line_flag = [-1]*len(segment_param_list);#标记每条线段从属的直线（线段组）序号
    line_index = -1;#直线序号
    segment_group_param_list = [];#线段组参数列表 参数如下：kmin,kmax,bmin,bmax,k_gatemin,k_gatemax,b_gatemin,b_gatemax,segmentNum,totalLen
    #比较线段，将满足相关条件的线段聚为一条直线（线段组）
    for index in range(len(segment_param_list)):
        if(line_flag[index] == -1):#判断该线段是否已经被合并
            #与现有直线进行合并判断
            for line_index in range(len(segment_group_param_list)):
                #合并判断
                flag1 = (segment_param_list[index][4] >= segment_group_param_list[line_index][4]) and                         (segment_param_list[index][4] <= segment_group_param_list[line_index][5]);
                flag2 = (segment_param_list[index][5] >= segment_group_param_list[line_index][6]) and                         (segment_param_list[index][5] <= segment_group_param_list[line_index][7]);
                if(flag1 and flag2):
                    line_flag[index] = line_index;#新线段融入新组
                    #更新线段组参数
                    if(segment_param_list[index][4] < segment_group_param_list[line_index][0]):
                        segment_group_param_list[line_index][0] = segment_param_list[index][4];
                    if(segment_param_list[index][4] > segment_group_param_list[line_index][1]):
                        segment_group_param_list[line_index][1] = segment_param_list[index][4];
                    if(segment_param_list[index][5] < segment_group_param_list[line_index][2]):
                        segment_group_param_list[line_index][2] = segment_param_list[index][5];
                    if(segment_param_list[index][5] < segment_group_param_list[line_index][3]):
                        segment_group_param_list[line_index][3] = segment_param_list[index][5];
                    
                    segment_group_param_list[line_index][4] = segment_group_param_list[line_index][0] - 0.3;
                    segment_group_param_list[line_index][5] = segment_group_param_list[line_index][1] + 0.3;
                    segment_group_param_list[line_index][6] = segment_group_param_list[line_index][2] - 100;
                    segment_group_param_list[line_index][7] = segment_group_param_list[line_index][3] + 100;
                    segment_group_param_list[line_index][8]+=1;
                    segment_group_param_list[line_index][9]+=segment_param_list[index][6];
                    break;#线段已合并入组，退出组循环
                    
            if(line_flag[index] == -1):#如果线段未能合并到任何组，则自创组
               line_index+=1;
               line_flag[index]=line_index;
               #创建新组的参数
               segment_group_param_list.append(
               [segment_param_list[index][4],
               segment_param_list[index][4],
               segment_param_list[index][5],
               segment_param_list[index][5],               
               segment_param_list[index][4] - 0.3,
               segment_param_list[index][4] + 0.3,
               segment_param_list[index][5] - 100,
               segment_param_list[index][5] + 100,
               1,
               segment_param_list[index][6]]);
               
    
    #测试用 绘出不同线段
    '''
    segment_image = np.copy(img);
    for index in range(len(segment_param_list)):
        x1 = segment_param_list[index][0];
        y1 = segment_param_list[index][1];
        x2 = segment_param_list[index][2];
        y2 = segment_param_list[index][3];
        
        if(line_flag[index] == 0):
            cv2.line(segment_image, (x1, y1), (x2, y2), [255,0,0], 5)
        elif(line_flag[index] == 1):
            cv2.line(segment_image, (x1, y1), (x2, y2), [0,255,0], 5)
        elif(line_flag[index] == 2):
            cv2.line(segment_image, (x1, y1), (x2, y2), [0,0,255], 5)
        elif(line_flag[index] == 3):
            cv2.line(segment_image, (x1, y1), (x2, y2), [255,0,255], 5)
        else:
            cv2.line(segment_image, (x1, y1), (x2, y2), [0,255,255], 5)
    plt.imshow(segment_image);
    '''

           
    #完成聚类后，统计每一线段组的x-y坐标，执行拟合
    line_param_list =[];#直线参数表 k,b,xmin,ymin,xmax,ymax
    for line_index in range(len(segment_group_param_list)):
        x_list=[];
        y_list=[];
        
        for index in range(len(segment_param_list)):
            if(line_flag[index] == line_index):#判断该线段是否属于当前类
               x1,y1,x2,y2,k,b,seg_length = segment_param_list[index];
               x_list.append(x1);                                                   
               x_list.append(x2);    
               y_list.append(y1);                                                   
               y_list.append(y2);    
               
               
        k, b = np.polyfit(x_list, y_list, 1);  
        #line_param_list.append((k,b,segment_group_param_list[line_index][9]));
        
        #对直线有效性进行判定
        if(segment_group_param_list[line_index][9] < 50):
            continue;
        elif(k > -0.85 and k < -0.4 and b > 400 and b < 800):
            line_param_list.append((k,b,segment_group_param_list[line_index][9]));
        elif(k > 0.4 and k < 0.8 and b > -200 and b < 200):
            line_param_list.append((k,b,segment_group_param_list[line_index][9]));
        else:
            continue;
        
    #直线间进行判断,剔除满足合并条件的目标
    line_valid_flag = [1]*len(line_param_list);
    for i in range(len(line_param_list)):
        k_1,b_1,len_1 = line_param_list[i];
        for j in range(i+1,len(line_param_list)):
            k_2,b_2,len_2 = line_param_list[j];
            if(math.fabs(k_2-k_1)<0.3 and math.fabs(b_2-b_1)<100):
                if(len_2>len_1):
                    line_valid_flag[i]=0;
                else:
                    line_valid_flag[j]=0;
            


    row,col,depth = img.shape;
    #计算每条直线的端点,绘图
    for i in range(len(line_param_list)):
        k,b,length = line_param_list[i];
        
        if(line_valid_flag[i]==0):#不绘制已被判定为无效的直线
            continue;
        
        y1 = (int)(row/2+50);
        x1 = (int)((y1-b)/k);
    

    #计算与图像下边界交点
        x=(int)((row-b)/k);
        if(x in range(img.shape[1])):#和下边界交点
            x2 = x;
            y2 = row;
        else:#和右边界交点
            x2 = col;
            y2 = (int)(k*x2+b);

        cv2.line(img, (x1, y1), (x2, y2), color, thickness);

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gama=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gama);


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[19]:

import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[20]:

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

#change to gray-scale
image_gray = grayscale(image);

plt.imshow(image_gray)  
plt.title('gray figure;')
plt.show()



# In[21]:

#Guass blur
kernel_size = 5
blur_gray = gaussian_blur(image_gray,kernel_size)

plt.imshow(blur_gray)
plt.title('blur figure;')
plt.show()


# In[22]:

#canny edge
low_threshold = 50
high_threshold = 150
edges = canny(blur_gray,low_threshold,high_threshold);
plt.imshow(edges,cmap='gray')
plt.title('road edge')
plt.show()


# In[23]:

# mask the edge(only keep the areas the lane lines may appear)

row,col = edges.shape;#get figure size
mask_area = np.array( [[[0,row-1],[int(col/2),int(row/2)+50],[col-1,row-1]]]);
edges_mask = region_of_interest(edges,mask_area)
plt.imshow(edges_mask,cmap='gray')
plt.title('road edge mask;')
plt.show()


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[24]:

rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 30
max_line_gap = 50

# Run Hough on edge detected image
lines = cv2.HoughLinesP(edges_mask, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap);

line_image = np.copy(image)  #creating a blank to draw lines on
draw_lines(line_image, lines)

plt.imshow(line_image)
plt.title('HoughLinesP result')
plt.show()


# In[25]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[26]:

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    #change to gray-scale
    image_gray = grayscale(image);
    
    #Guass blur
    kernel_size = 5
    blur_gray = gaussian_blur(image_gray,kernel_size)

    #canny edge
    low_threshold = 100
    high_threshold = 150
    edges = canny(blur_gray,low_threshold,high_threshold);
    
    # mask the edge(only keep the areas the lane lines may appear)
    row,col = edges.shape;#get figure size
    mask_area = np.array( [[[0,row-1],[int(col/2),int(row/2)+50],[col-1,row-1]]]);
    edges_mask = region_of_interest(edges,mask_area)
 
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 20
    max_line_gap = 20

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges_mask, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap);

    line_image = np.zeros_like(image)  #creating a blank to draw lines on
    draw_lines(line_image, lines,[255,0,0],10);
    
    result_img = weighted_img(line_image, image, 1, 0.8);

    return result_img;


# Let's try the one with the solid white lane on the right first ...

# In[27]:

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[28]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[29]:

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[30]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:

challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

