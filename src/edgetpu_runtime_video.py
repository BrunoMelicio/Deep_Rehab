import argparse
import time
import cv2
import numpy as np

from PIL import Image

import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def input_details(interpreter, key):
  """Returns input details by specified key."""
  return interpreter.get_input_details()[0][key]

def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = input_details(interpreter, 'index')
  return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:, :] = data

def set_input2(interpreter, data):
  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Inputs for the TFLite model must be uint8, so we quantize our input data.
  # NOTE: This step is necessary only because we're receiving input data from
  # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
  # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
  #   input_tensor[:, :] = data
  scale, zero_point = input_details['quantization']
  input_tensor[:, :] = np.uint8(data / scale + zero_point)


interpreter = make_interpreter("converted/deeprehab_edgetpu.tflite")
interpreter.allocate_tensors()

_, height, width, _ = interpreter.get_input_details()[0]['shape']

#image = Image.open("couple.jpg").convert('RGB').resize((width, height), Image.ANTIALIAS)

video = cv2.VideoCapture('test_video.mp4')

fpss = []
inferences = []

while(video.isOpened()):

    ret, frame = video.read()

    if ret == False:
        break

    frame = cv2.resize(frame, (224,224), interpolation= cv2.INTER_CUBIC)
    img = frame.astype(np.float32) #uint8
    #img = np.expand_dims(img, axis=0)

    set_input(interpreter, img)

    #input_tensor= np.array(np.expand_dims(img,0), dtype=np.float16)

    start = time.time()

    #interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    #output_data = interpreter.get_tensor(output_details[0]['index'])

    #print(output_data)

    end = time.time()

    seconds = end - start
    inferences.append(seconds*1000)

    fps  = 1 / seconds
    fpss.append(fps)
    #print("Estimated frames per second : {0}".format(fps))

    #print("All:",len(pred_kp))

    #draw_kps(selected, img)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()

print("Average FPS:", sum(fpss)/len(fpss))
print("Average Inference:", sum(inferences)/len(inferences))

'''
set_input(interpreter, image)

print('----INFERENCE TIME----')
print('Note: The first inference on Edge TPU is slow because it includes',
    'loading the model into Edge TPU memory.')
for _ in range(5):
  start = time.perf_counter()
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  #classes = classify.get_output(interpreter, args.top_k, args.threshold)
  print('%.1fms' % (inference_time * 1000))

'''