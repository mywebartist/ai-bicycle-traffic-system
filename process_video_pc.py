import cv2
import numpy as np
import tensorflow as tf
import time
import datetime as dt

 

# system variables
res = 512
modelpath = '11bicycle_traffic_management_system_512.h5'
videofile = 'VID_20220612_163050_testf_512.mp4'
nowts = time.time()
result = 0
secondsToAddOnDetection = 5
showPreview = True

model = tf.keras.models.load_model(
    modelpath, custom_objects=None, compile=True, options=None
)

def check_frame(img):
  img = np.array([img])
  img = img/255
  result = model.predict(img, verbose=0)
  result = int( np.round( result[0][0]))
  #print(result)
  return result

def process_video():
  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  greentime = dt.datetime.now()
  cap = cv2.VideoCapture(videofile)
  source_vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  source_vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

  print('starting')
  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    
    # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
  
      # reduec resolution
      if source_vid_height != res or source_vid_width != res:
        frame = cv2.resize(frame, (res, res))
      
            # Display the resulting frame
      if showPreview:
        cv2.imshow('Bicycle Traffic Manage System',frame)

      result = check_frame(frame)
      if result == 1:
        print('Saw bike. Turn green.')
        bike = 'Saw bike. ' 
        greentime = dt.datetime.now()+ dt.timedelta(0,secondsToAddOnDetection)
      else:
        message = 'Red light'
      

      if ( greentime - dt.datetime.now()).total_seconds() > 0:
        # print('im gogo')
        message = bike + 'Green light count down: ' + str( round( ( greentime - dt.datetime.now()).total_seconds() ) )

      print( message)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  
  
  # When everything done, release the video capture object
  cap.release()
  
  # Closes all the frames
  cv2.destroyAllWindows()



#load_model()
process_video()
