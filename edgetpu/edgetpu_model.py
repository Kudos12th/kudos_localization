import time
import os
import logging

import yaml
import numpy as np
import pycoral.utils.edgetpu as etpu
from pycoral.adapters import common

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeTPUModel")

class EdgeTPUModel:

    def __init__(self, model_file):
        """
        Inputs:
          - model_file: path to edgetpu-compiled tflite file
        """
        model_file = os.path.abspath(model_file)
        if not model_file.endswith('tflite'):
            model_file += ".tflite"
            
        self.model_file = model_file

        self.inference_time = None
        self.interpreter = None
        
        self.make_interpreter()
        self.get_image_size()
        

    def get_names(self, path):
        """
        Load a names file
        
        Inputs:
          - path: path to names file in yaml format
        """

        with open(path, 'r') as f:
          cfg = yaml.load(f, Loader=yaml.SafeLoader)
          
        names = cfg['names']
        logger.info("Loaded {} classes".format(len(names)))
        
        self.names = names
    

    def make_interpreter(self):
        """
        Internal function that loads the tflite file and creates
        the interpreter that deals with the EdgetPU hardware.
        """
        # Load the model and allocate
        self.interpreter = etpu.make_interpreter(self.model_file)
        self.interpreter.allocate_tensors()
    
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.debug(self.input_details)
        logger.debug(self.output_details)
        
        self.input_zero = self.input_details[0]['quantization'][1]
        self.input_scale = self.input_details[0]['quantization'][0]
        self.output_zero = self.output_details[0]['quantization'][1]
        self.output_scale = self.output_details[0]['quantization'][0]
        
        # If the model isn't quantized then these should be zero
        # Check against small epsilon to avoid comparing float/int
        if self.input_scale < 1e-9:
            self.input_scale = 1.0
        
        if self.output_scale < 1e-9:
            self.output_scale = 1.0
    
        logger.debug("Input scale: {}".format(self.input_scale))
        logger.debug("Input zero: {}".format(self.input_zero))
        logger.debug("Output scale: {}".format(self.output_scale))
        logger.debug("Output zero: {}".format(self.output_zero))
        
        logger.info("Successfully loaded {}".format(self.model_file))
    

    def get_image_size(self):
        """
        Returns the expected size of the input image tensor
        """
        if self.interpreter is not None:
            self.input_size = common.input_size(self.interpreter)
            logger.debug("Expecting input shape: {}".format(self.input_size))
            return self.input_size
        else:
            logger.warn("Interpreter is not yet loaded")
        
    
    def forward(self, x:np.ndarray, with_nms=True) -> np.ndarray:
        
        tstart = time.time()
        # Transpose if C, H, W
        if x.shape[0] == 3:
          x = x.transpose((1,2,0))
        
        x = x.astype('float32')

        # Scale input, conversion is: real = (int_8 - zero)*scale
        x = (x/self.input_scale) + self.input_zero
        x = x[np.newaxis].astype(np.uint8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        
        # Scale output
        result = (common.output_tensor(self.interpreter, 0).astype('float32') - self.output_zero) * self.output_scale
        self.inference_time = time.time() - tstart
        
        return result
          