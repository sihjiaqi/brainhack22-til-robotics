from fileinput import filename
from typing import List, Any
from tilsdk.cv.types import *
import onnxruntime as ort
from tensorflow import keras

class CVService:
    def __init__(self, model_dir):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        self.CVModel = keras.models.load_model(model_dir)
        self.id = 0
        # TODO: Participant to complete.

    def targets_from_image(self, img) -> List[DetectedObject]:
        '''Process image and return targets.
        
        Parameters
        ----------
        img : Any
            Input image.
        
        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        CVPrediction = self.CVModel(img)
        
        # loop through each obj found in an image
        for item in range(len(CVPrediction[0]['boxes'])):
            self.id += 1
            boxes =  list(CVPrediction[0]['boxes'][item])

            # format: BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'w', 'h'])
            bbox = BoundingBox(boxes[0], boxes[1], boxes[2]-boxes[0], boxes[3]-boxes[1])
            
            # DetectedObject = namedtuple('DetectedObject', ['id', 'cls', 'bbox'])
            obj = DetectedObject(self.id, int(CVPrediction[0]['labels'][item]), bbox)
        
        return [obj]
        # TODO: Participant to complete.


class MockCVService:
    '''Mock CV Service.
    
    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        # Does nothing.
        pass

    def targets_from_image(self, img:Any) -> List[DetectedObject]:
        '''Process image and return targets.
        
        Parameters
        ----------
        img : Any
            Input image.
        
        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        # dummy data
        bbox = BoundingBox(100,100,300,50)
        obj = DetectedObject("1", "1", bbox)
        return [obj]