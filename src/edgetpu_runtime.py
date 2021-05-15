import argparse
import time

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


interpreter = make_interpreter("converted/deeprehab_edgetpu.tflite")
interpreter.allocate_tensors()

_, height, width, _ = interpreter.get_input_details()[0]['shape']

image = Image.open("test.jpg").convert('RGB').resize((width, height), Image.ANTIALIAS)

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