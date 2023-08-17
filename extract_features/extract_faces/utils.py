''' 
This script has functions that extract face presence and location from a movie.

This script requires a .json file that has information about the movie, path locations, and the desired sampling rate from the movie.
'''

# import modules
import imageio
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pliers
import face_recognition
import os
from os.path import join
import json

from pliers.stimuli import VideoStim
from pliers.graph import Graph
from pliers.filters import FrameSamplingFilter
# from pliers.extractors import (FaceRecognitionFaceLocationsExtractor,
#                                FaceRecognitionFaceEncodingsExtractor,
#                                MicrosoftAPIFaceExtractor,
#                                GoogleVisionAPIFaceExtractor,
#                                merge_results)
from pliers.extractors import (FaceRecognitionFaceLocationsExtractor,
                               FaceRecognitionFaceEncodingsExtractor,
                               merge_results)

def extract_faces(json_filepath):
    '''
    Extracts when and where faces are present in a movie clip, saved out as a df.
    
    Parameters
    ----------
    json_filepath: the .json file that houses all information to be passed to function
    
    '''
    # # Open feature json file
    # with open(json_filepath) as f:
    #     data = json.load(f)
        
    # Opening JSON file
    f = open('feature_AHH.json')
    # returns JSON object as a dictionary
    data = json.load(f)
        
    # load in relevant json data
    for stimuli, stimuli_data in data.items():
        dir = stimuli_data['dir']
        samplerate = stimuli_data['samplerate']
        downloadpath = stimuli_data['downloadpath']
        movies = stimuli_data['movies']
        savepath = stimuli_data['savepath']
        
        # For each movie, load in data and extract faces
        for movie in movies:
            moviepath = os.path.join(downloadpath, movie)
            movienoextension = movie[:len(movie)-4]
            # data = np.load(savepath + movienoextension + "_gray.npz", allow_pickle=True)['movie']
            
            video = VideoStim(moviepath)
            
            sampler = FrameSamplingFilter(hertz=samplerate)
            frames = sampler.transform(video)
            
            # Detect faces in selected frames
            face_ext = FaceRecognitionFaceLocationsExtractor()
            face_result = face_ext.transform(frames)
            
            # Save out results
            result_df = [f.to_df() for f in face_result]
            result_df = pd.concat(result_df)
            
            # Show what's there...
            result_df.head(10)
            
            np.savez(savepath + movienoextension + '_faces.npz', features=result_df)
            # np.save(savepath + movienoextension + '_faces.npz')
            
            # result_df.to_csv('FaceExtractor_' + movienoextension + '.csv', index = False)

