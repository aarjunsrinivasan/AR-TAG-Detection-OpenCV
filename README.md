# ENPM 673 - Project 1

The project focuses on detecting a custom AR Tag (a form of feducial marker), that is used for obtaining a
point of reference in the real world, such as in augmented reality applications. The two aspects to using an
AR Tag: detection and tracking, has been implemented in this project. Following are the 2 stages:
- Detection: Involves finding the AR Tag from a given frame in the video sequence
- Tracking: Involves keeping the tag in \view" throughout the sequence and performing image processing
operations based on the tag's orientation and position (a.k.a. the pose).
After detection and tracking of AR tags, we perform two tasks, namely superimposing Lena image, and
placing a virtual cube over the AR tag.

## Packages Required
- NumPy
- Matplotlib
- OpenCV
- Math

## To run the codes
- Run the python files in the current directory which contains all the codes.
- The code runs  with the sample video "Tag0.mp4" placed in the Data folder
- Place the relative path of the video you want to run in,cap = cv2.VideoCapture('Data/Tag0.mp4')
- Open terminal run python3 ARtag.py
 
