from typing import Iterable, List
import base64
from tilsdk.localization.types import *
import onnxruntime as ort
import librosa
import numpy as np

class NLPService:
    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        self.sess = ort.InferenceSession("C:\Users\joswong\OneDrive - NVIDIA Corporation\Documents\GitHub\brainhack22_robotics\model\nlp_model.onnx", providers=["CUDAExecutionProvider"])
        self.input_name1 = self.sess.get_inputs()[0].name
        self.input_name2 = self.sess.get_inputs()[1].name
        self.output_name = self.sess.get_outputs()[0].name
        self.classes = ['distress', 'good']
        # TODO: Participant to complete.
        pass

    def locations_from_clues(self, clues:Iterable[Clue]) -> List[RealLocation]:
        '''Process clues and get locations of interest.
        
        Parameters
        ----------
        clues
            Clues to process.

        Returns
        -------
        lois
            Locations of interest.
        '''
        
        mfcc_data ,melspec_data = self.decode_audiostring(clues.audio)

        sess_pred = self.sess.run([self.output_name], {self.input_name1: mfcc_data.astype(np.float64), self.input_name2:melspec_data.astype(np.float64)})[0]
        pred = self.classes[np.unravel_index(np.argmax(sess_pred, axis=None), sess_pred.shape)]

        if pred == 'distress':
            locations = [c.location for c in clues]
        elif pred == 'good':
            locations = []

        return locations

    def convert_to_mfcc(self, audio):
        mfccs = librosa.feature.mfcc(audio, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13)

        if mfccs.shape[1] < 151:
            result = np.zeros((13,151 - mfccs.shape[1]),dtype=float)
            new_mfcc = np.hstack((mfccs,result))
            return new_mfcc
        else:
            return mfccs

    def convert_to_melspec(self, audio):
        mel_spec = librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
        S_DB = librosa.power_to_db(mel_spec, ref=np.max)

        if S_DB.shape[1] < 151:
            result = np.zeros((256,151 - S_DB.shape[1]),dtype=float)
            new_melspec = np.hstack((S_DB,result))
            return new_melspec
        else:
            return S_DB

    def decode_audiostring(self, audiostring):
        wav_file = open("data/audio/temp.wav", "wb")
        decode_string = base64.b64decode(audiostring)
        wav_file.write(decode_string)

        audio_file = "data/audio/temp.wav"
        audio, sample_rate = librosa.load(audio_file, sr=22050)
        return self.convert_to_mfcc(audio), self.convert_to_melspec(audio)



class MockNLPService:
    '''Mock NLP Service.
    
    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        pass

    def locations_from_clues(self, clues:Iterable[Clue]) -> List[RealLocation]:
        '''Process clues and get locations of interest.
        
        Mock returns location of all clues.
        '''
        locations = [c.location for c in clues]

        return locations