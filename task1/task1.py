import numpy as np
import cv2 as cv
import os


# some function to work with a frame that will be used to find a blue pen
def find_blue_pixels_straight(frame):
    # working with a copy of a frame
    # assigning more bit to a pixel for comparison
    work_frame = np.uint16(frame.copy())
    # making a binary frame
    #blue pixels
    work_frame[~((work_frame[:, :, 0] > work_frame[:, :, 1] + 70) & (work_frame[:, :, 0] > work_frame[:, :, 2] + 70))] = (0, 0, 0)
    # others as black
    work_frame[((work_frame[:, :, 0] > work_frame[:, :, 1] + 70) & (work_frame[:, :, 0] > work_frame[:, :, 2] + 70))] = (255, 255, 255)
    # turn it back ot uint8
    return np.uint8(work_frame) 

# not successful one
# def find_blue_pixels_hsv(frame):
#     work_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2HSV)
#     h, s, v = cv.split(work_frame)
#     s.fill(255)
#     v.fill(255)
#     hsv_image = cv.merge([h, s, v])
#     lower = np.array([117, 255, 255], dtype = np.uint8)
#     upper = np.array([127, 255, 255], dtype = np.uint8)
    
#     return cv.inRange(hsv_image, lower, upper)

# main part
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (sq tream end?). Exiting ...")
        break

    # finding blue pixels in 
    work_frame = find_blue_pixels_straight(frame)
    #work_frame2  = find_blue_pixels_hsv(frame)
    contour_frame = cv.cvtColor(work_frame.copy(), cv.COLOR_BGR2GRAY)
    #contour_frame = cv.equalizeHist(contour_frame)
    # to stabilize an image - making some blur
    # from here: https://stackoverflow.com/questions/71739517/detect-squares-paintings-in-images-and-draw-contour-around-them-using-python
    contour_frame = cv.GaussianBlur(contour_frame, (17,17), 0)
    # as an image already binery tbh, we'll treshhold it again
    ret, thresh = cv.threshold(contour_frame, 0, 255, cv.THRESH_BINARY)
    # looking for contours of an object
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # creating a rectangle of a contour and displaying it
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(work_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('frame', np.hstack((frame, work_frame)))
    #cv.imshow('frame1', work_frame2)
    
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()