import cv2

#loading some pre-set data from opencv
trained_face_data = cv2.CascadeClassifier('E:\AI & ML\FaceDetector app\haarcascade_frontalface_default.xml')

# webcam setup for capturing video
webcam = cv2.VideoCapture(0)

# always true while loop for real-time face detection
while True:

    #reading frame
    sucessful_frame_read, frame = webcam.read()
    
    #converting that frame into grey-scaled image
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # extracting frame's face coordinates
    face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

    # drawing a rectangle around that frame using coordinates
    for(x, y, w, h) in face_coordinates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0),2)
    
    # displaying that frame 
    cv2.imshow("show face", frame)

    # wait time is 1 milli seconds, 'Q' key is assigned to terminate the app
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
          break
webcam.release()