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
from joblib import load

class NLPService:
    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        self.sess = ort.InferenceSession(model_dir, providers=["CUDAExecutionProvider"])
        self.mfcc_scaler = load('/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/NLP_MFCC_MMScaler.bin')
        self.melspec_scaler = load('/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/NLP_MelSpec_MMScaler.bin')
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

        sess_pred = self.sess.run([self.output_name], {self.input_name1: mfcc_data.astype(np.float64), self.input_name2:melspec_data.astype(np.float64)})[0]
        pred = self.classes[np.argmax(sess_pred)]
        print(pred)
        pred = 'distress'
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
            return self.normalise_mfcc(new_mfcc)
        else:
            return self.normalise_mfcc(mfccs)

    def convert_to_melspec(self, audio):
        mel_spec = librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
        S_DB = librosa.power_to_db(mel_spec, ref=np.max)

        if S_DB.shape[1] < 151:
            result = np.zeros((256,151 - S_DB.shape[1]),dtype=float)
            new_melspec = np.hstack((S_DB,result))
            return self.normalise_melspec(new_melspec)
        else:
            return self.normalise_melspec(S_DB)

    def normalise_mfcc(self, mfcc):
        mfcc_reshape = mfcc[np.newaxis,:, :]
        mfcc_temp = mfcc.reshape(mfcc_reshape.shape[0], mfcc_reshape.shape[1]*mfcc_reshape.shape[2])
        norm_mfcc = self.mfcc_scaler.transform(mfcc_temp).reshape(mfcc_reshape.shape[0], mfcc_reshape.shape[1], mfcc_reshape.shape[2])
        return norm_mfcc

    def normalise_melspec(self, melspec):
        melspec_reshape = melspec[np.newaxis,:, :]
        melspec_temp = melspec.reshape(melspec_reshape.shape[0], melspec_reshape.shape[1]*melspec_reshape.shape[2])
        norm_melspec = self.melspec_scaler.transform(melspec_temp).reshape(melspec_reshape.shape[0], melspec_reshape.shape[1], melspec_reshape.shape[2])
        return norm_melspec


    def decode_audiostring(self, audiostring):
        data, rate = sf.read(io.BytesIO(audiostring))
        data = librosa.resample(data, orig_sr=rate, target_sr=22050)

        return self.convert_to_mfcc(data), self.convert_to_melspec(data)



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
        self.sess = ort.InferenceSession(model_dir, providers=["CUDAExecutionProvider"])
        self.mfcc_scaler = load('/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/NLP_MFCC_MMScaler.bin')
        self.melspec_scaler = load('/mnt/c/Users/user/Documents/GitHub/brainhack22_robotics/model/NLP_MelSpec_MMScaler.bin')
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

        sess_pred = self.sess.run([self.output_name], {self.input_name1: mfcc_data.astype(np.float64), self.input_name2:melspec_data.astype(np.float64)})[0]
        pred = self.classes[np.argmax(sess_pred)]
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
            return self.normalise_mfcc(new_mfcc)
        else:
            return self.normalise_mfcc(mfccs)

    def convert_to_melspec(self, audio):
        mel_spec = librosa.feature.melspectrogram(audio, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
        S_DB = librosa.power_to_db(mel_spec, ref=np.max)

        if S_DB.shape[1] < 151:
            result = np.zeros((256,151 - S_DB.shape[1]),dtype=float)
            new_melspec = np.hstack((S_DB,result))
            return self.normalise_melspec(new_melspec)
        else:
            return self.normalise_melspec(S_DB)

    def normalise_mfcc(self, mfcc):
        mfcc_reshape = mfcc[np.newaxis,:, :]
        mfcc_temp = mfcc.reshape(mfcc_reshape.shape[0], mfcc_reshape.shape[1]*mfcc_reshape.shape[2])
        norm_mfcc = self.mfcc_scaler.transform(mfcc_temp).reshape(mfcc_reshape.shape[0], mfcc_reshape.shape[1], mfcc_reshape.shape[2])
        return norm_mfcc

    def normalise_melspec(self, melspec):
        melspec_reshape = melspec[np.newaxis,:, :]
        melspec_temp = melspec.reshape(melspec_reshape.shape[0], melspec_reshape.shape[1]*melspec_reshape.shape[2])
        norm_melspec = self.melspec_scaler.transform(melspec_temp).reshape(melspec_reshape.shape[0], melspec_reshape.shape[1], melspec_reshape.shape[2])
        return norm_melspec


    def decode_audiostring(self, audiostring):
        data, rate = sf.read(io.BytesIO(audiostring))
        data = librosa.resample(data, orig_sr=rate, target_sr=22050)

        return self.convert_to_mfcc(data), self.convert_to_melspec(data)