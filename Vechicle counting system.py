#!!!vechicle counting system using deeplearning!!!
import cv2
import numpy as np


# web camera
cap = cv2.VideoCapture('video.mp4')

min_width_react = 80 #min width bounded rectangle around the detected vechicle
min_height_react = 80 #min height bounded rectangle around the detected vechicle
count_line_position = 550
# Initialize Substructor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()#for detecting moving object 



def center_handle(x,y,w,h):#Making a circle to all for counting vechicle.
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = x+x1
    return cx,cy

detect = [] # list to store the coordinate of vechicle center
offset = 6  #Allowable error between pixel detectint vechicle crossing the counting line
counter = 0 #count to keep track the no of vechicle



#main loop processing each frame of the video
while True:
    ret,frame1 = cap.read()
    grey = cv2.cvtcolor(frame1,cv2.COLOR_BGR2GRAY)#convert the gray into Gaussian blur for noise reduction
    blur = cv2.GaussianBlur(grey,(3,3),5)
    # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones(5,5))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))#giving a structure for further process
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernal)#applying morphological operation(creating a output image as same size)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernal)#helps to providing a same shape in output
    countershape,h = cv2.findContours(dilatada, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#helps to count the vechicle
    
    cv2.line(frame1,(25,count_line_position),(1200, count_line_position),(225,127,0),3)
    for (i,c) in enumerate(countershape):
        (x,y,w,h) = cv2.boundingReact(c)
        validate_counter = (w >= min_width_react) and (w >= min_height_react)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"VECHICLE COUNTER: " +str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,225),5)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position + offset) and y>(count_line_position - offset):
                counter += 1
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                detect.remove((x,y))

            print("vechicle counter:"+str(counter))


    cv2.putText(frame1,"VECHICLE COUNTER: " +str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,225),5)


    # cv2.imshow('Detector',dilatada)
    # cv2.imshow('Video Original',frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()