import cv2
import numpy as np
from random import randint
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time
import datetime as dt


#led gpio connection
LED_RED = 16
LED_YELLOW = 17

#setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_RED, GPIO.OUT)
GPIO.setup(LED_YELLOW, GPIO.OUT)

#turn off lights
GPIO.output(LED_YELLOW, GPIO.LOW)
GPIO.output(LED_RED, GPIO.LOW)

# system variables
res = 512
modelpath = '11bicycle_traffic_management_system_512.tflite'
videofile = 'VID_20220612_163050_testf_512.mp4'
nowts = time.time()
result = 0
secondsToAddOnDetection = 5
showPreview = True

def check_frame(img):
  input_data = np.array(img[np.newaxis,:,:,:] / 255, dtype=np.float32)
  #print(input_data.shape)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  
  interpreter.invoke()
  
  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  result = interpreter.get_tensor(output_details[0]['index'])
  result = int(np.round( result[0][0]))
  return result


## setup model
# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=modelpath)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']

def load_model():
  ## setup model
  # Load the TFLite model and allocate tensors.
  interpreter = tflite.Interpreter(model_path=modelpath)
  print( type(interpreter))
  interpreter.allocate_tensors()
  
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  # Test the model on random input data.
  input_shape = input_details[0]['shape']
  
  ##############



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
        #bike = '' 
        greentime = dt.datetime.now()+ dt.timedelta(0,secondsToAddOnDetection)
      else:
        message = 'Red light'
      

      if ( greentime - dt.datetime.now()).total_seconds() > 0:
        # print('im gogo')
        message =  'Green light count down: ' + str( round( ( greentime - dt.datetime.now()).total_seconds() ) )
        # set lights
  
        GPIO.output(LED_RED, GPIO.LOW)
        GPIO.output(LED_YELLOW, GPIO.HIGH)
      else:
        GPIO.output(LED_RED, GPIO.HIGH)
        GPIO.output(LED_YELLOW, GPIO.LOW)
 
      # print('random' + str( randint(5,10) ) )
 
      # print(result)
      print( message)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  
  
  ## turn off lights
  GPIO.output(LED_RED, GPIO.LOW)
  GPIO.output(LED_YELLOW, GPIO.LOW)
  
  # When everything done, release the video capture object
  cap.release()
  
  # Closes all the frames
  cv2.destroyAllWindows()



#load_model()
process_video()
