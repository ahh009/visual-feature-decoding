'''
Feature extraction: extract faces
    Use the face_recognition package in the pliers package to for automated extraction of features from .mp4 data.
    
'''

# Install pliers_extractors
from pliers.extractors import FaceRecognitionFaceLocationsExtractor
from os.path import join
from pliers.tests.utils import get_test_data_path

# A test picture
image = ()

# Initialize extractor
ext = FaceRecognitionFaceLocationsExtractor()








