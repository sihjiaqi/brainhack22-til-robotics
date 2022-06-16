from email.mime import audio
from typing import Iterable, List
import base64
from tilsdk.localization.types import *
import onnxruntime as ort
import librosa
import numpy as np
import io
from scipy.io.wavfile import read, write
import soundfile as sf

class NLPService:
    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        self.sess = ort.InferenceSession("/home/joshua/Documents/GitHub/brainhack22_robotics/model/nlp_model.onnx", providers=["CUDAExecutionProvider"])
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
        
        audioclue = [c.audio for c in clues]
        mfcc_data ,melspec_data = self.decode_audiostring(audioclue[0])
        sess_pred = self.sess.run([self.output_name], {self.input_name1: mfcc_data.astype(np.float64), self.input_name2:melspec_data.astype(np.float64)})[0]
        pred = self.classes[np.unravel_index(np.argmax(sess_pred, axis=None), sess_pred.shape)]
        print(pred)
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
        self.sess = ort.InferenceSession("/home/joshua/Documents/GitHub/brainhack22_robotics/model/nlp_model.onnx", providers=["CUDAExecutionProvider"])
        self.input_name1 = self.sess.get_inputs()[0].name
        self.input_name2 = self.sess.get_inputs()[1].name
        self.output_name = self.sess.get_outputs()[0].name
        self.classes = ['distress', 'good']

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
        
        audioclue = [c.audio for c in clues]
        mfcc_data ,melspec_data = self.decode_audiostring(audioclue[0])
        print(mfcc_data.shape)
        print(melspec_data.shape)


        mfcc_reshape = mfcc_data[np.newaxis,:, :]
        mel_spec_reshape = melspec_data[np.newaxis,:, :]

        sess_pred = self.sess.run([self.output_name], {self.input_name1: mfcc_reshape.astype(np.float64), self.input_name2:mel_spec_reshape.astype(np.float64)})[0]
        pred = self.classes[np.argmax(sess_pred)]
        print(pred)
        if pred == 'distress':
            locations = [c.location for c in clues]
        elif pred == 'good':
            locations = []

        return locations

    def convert_to_mfcc(self, audio):
        mfccs = librosa.feature.mfcc(audio, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13)

        print(mfccs.shape)

        if mfccs.shape[1] < 151:
            result = np.zeros((13, 151-mfccs.shape[1]),dtype=float)
            new_mfcc = np.hstack((mfccs,result))
            return new_mfcc
        else:
            return mfccs

    def convert_to_melspec(self, audio):
        mel_spec = librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
        S_DB = librosa.power_to_db(mel_spec, ref=np.max)
        print(S_DB.shape)
        if S_DB.shape[1] < 151:
            result = np.zeros((256, 151 - S_DB.shape[1]),dtype=float)
            new_melspec = np.hstack((S_DB,result))
            return new_melspec
        else:
            return S_DB

    def decode_audiostring(self, audiostring):
        # audio, sr = librosa.load(audiostring, sr=22050)
        print(type(audiostring))
        print(len(audiostring))
        data, rate = sf.read(io.BytesIO(audiostring))
        data = librosa.resample(data, orig_sr=rate, target_sr=22050)

        return self.convert_to_mfcc(data), self.convert_to_melspec(data)