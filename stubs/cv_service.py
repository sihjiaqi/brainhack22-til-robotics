from fileinput import filename
from typing import List, Any
from tilsdk.cv.types import *
import onnxruntime as ort
# import onnx
import torchvision.transforms as transforms
import cv2


class CVService:
    def __init__(self, model_dir):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        self.id = 0
        # load ONNX model
        self.model_path = "/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/cv_model.onnx"
        # self.onnx_model = onnx.load(self.model_path)
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

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
        obj_arr = []
        # Process image and detect targets
        # convert numpy to tensor
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(img)
        tensor_img = tensor_img.unsqueeze_(0)

        # make prediction
        result = self.session.run(None, {'input': tensor_img.numpy()})
        
        # loop through each obj found in an image
        for item in range(len(result[0])):
            self.id += 1
            box =  list(result[0][item])

            # format: BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'w', 'h'])
            x_center = (box[2] - box[0])/ 2
            y_center = (box[3] - box[1])/ 2
            bbox = BoundingBox(x_center.astype(float), y_center.astype(float), (box[2]-box[0]).astype(float), (box[3]-box[1]).astype(float))
            print(type(bbox[2]-box[0]))
            # DetectedObject = namedtuple('DetectedObject', ['id', 'cls', 'bbox'])
            #print(self.id, int(result[1][item]), bbox)

            obj_arr.append(DetectedObject(int(self.id), int(result[1][item]), bbox))
            #print(obj_arr)
        return obj_arr
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