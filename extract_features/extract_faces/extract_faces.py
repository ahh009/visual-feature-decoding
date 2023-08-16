'''
Feature extraction: extract faces
    Use the face_recognition package in the pliers package to for automated extraction of features from .mp4 data.

Setup for using pliers to extract faces from .mp4 files:
    1. Install pliers and necessary dependencies
    2. Ensure video files to be extracted from are in the correct folder (filepath currently hardcoded)
    
TODO:
    - Create function to auto populate file path?
    - Create function to auto detect video length (in s)?
    - Figure out what result_df is actually storing...
'''

# If pliers + face_recognition not currently installed:
pip install pliers
pip install face_recognition

# import packages
import imageio
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pliers
from os.path import join

from pliers.stimuli import VideoStim
from pliers.graph import Graph
from pliers.filters import FrameSamplingFilter
#from pliers.extractors import (FaceRecognitionFaceLocationsExtractor,
                               #FaceRecognitionFaceEncodingsExtractor,
                               #MicrosoftAPIFaceExtractor,
                               #GoogleVisionAPIFaceExtractor,
                               #merge_results)
from pliers.extractors import (FaceRecognitionFaceLocationsExtractor,
                               FaceRecognitionFaceEncodingsExtractor,
                               merge_results)

from pliers.converters import VideoToAudioConverter


# Save the video to be analyzed into a variable.
video = VideoStim(r'/home/jovyan/hackathon/visual-feature-decoding/extract_features/extract_faces/video_clips/Discussion Stock Footage - Discussion Free Stock Videos - Discussion No Copyright Videos (480p).mp4')
#    AHH: This is currently hardcoded to my jupyterhub file path where the video to analyze is saved.
#    Fix: create a function that will substitute the first part of the path string?

# This is sampling at the rate of 2 Hz (2 frames per sec). To downsample further, change hertz to 1 or 0.5. 
#    AHH: This step takes a while, but is quicker the more you downsample. 
sampler = FrameSamplingFilter(hertz=2)
frames = sampler.transform(video)

# Detect faces in selected frames
face_ext = FaceRecognitionFaceLocationsExtractor()
# face_ext = FaceRecognitionFaceEncodingsExtractor() # AHH - need to figure out how EncodingsExtractor differs from LocationsExtractor
face_result = face_ext.transform(frames)


# AHH: Let's just look at the df...
result_df = [f.to_df() for f in face_result]

result_df = pd.concat(result_df)

result_df.head(10)


import numpy as np
# Set the range of time over which to plot extracted faces.
#     AHH: Again, this needs to be manually set currently. Maybe set the range automatically based on the number of frames given the sampling Hz...?
#     Total s = number of frames / Hz
time = np.arange(0,342,0.5)


# AHH: I'm pretty sure all this is doing is plotting whether a face is present or not at each sampled frame.
# Plotting location or face_encoding could be more interesting!
plt.plot(time, [t in result_df["onset"].values for t in time])


