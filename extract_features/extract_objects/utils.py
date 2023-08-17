# utils for semantic segmentation 

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import subprocess
from skimage import io
from skimage import img_as_ubyte
import imageio
import numpy as np
import skvideo 
skvideo.setFFmpegPath("/srv/conda/envs/lowlevel/bin")
import skvideo.io 
from moviepy.editor import VideoFileClip
from skimage.transform import resize
from tqdm import tqdm
import PIL.Image as pil_image
from joblib import Parallel, delayed
from time import time
from scipy.signal import convolve
from scipy import signal
import _pickle as pickle
import torch
from torchvision import transforms
from torchvision import models

fcn = models.segmentation.fcn_resnet101(weights=True).eval()

def preprocess_image(rgb_image):
    ''' Applies the following preprocessing steps to the RGB image:
    - Transforms image to a tensor with range [0, 1] and dimensions (nimg, 3, vdim, hdim) where nimg = number of images, vdim = height in pixels, and hdim = width in pixels
    - Normalizes values using ImageNet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

    Parameters
    ----------
    rgb_image : RGB image array

    Returns
    -------
    input_tensor : preprocessed, input-ready RGB image
    '''
    preprocess = transforms.Compose([
        #transforms.Resize(size=(hdim, vdim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(rgb_image)
    input_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_tensor

def movie_to_object_matrix(json_filepath):
    ''' Iterates over movies in movie list and , and saves it out as an .npz. 
    
    Parameters
    ----------
    json_filepath : the .json file that houses all information to be passed to function
    
    Returns
    -------
    objects_in_vid : a (21, Nframes) boolean matrix indicating which objects were present in each frame sampled from the movie. The pre-trained FCN has been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset (not including background, which comprises 1 of the 21 listed objects).
    
    '''
    # Open feature json file
    with open(json_filepath) as f:
        data = json.load(f)

    # Isolate all info for each stimuli
    for stimuli, stimuli_data in data.items():
        downloadpath = stimuli_data['downloadpath']
        movies = stimuli_data['movies']
        savepath = stimuli_data['savepath']
        
        # For each movie in movie list
        for movie in movies:
            
            moviepath = os.path.join(downloadpath, movie)
            movienoextension = movie[:len(movie)-4]
            if not os.path.exists(savepath + movienoextension + '_gray.npz'):
            
                # Open the video file using VideoFileClip
                video = VideoFileClip(moviepath)
            
                # Extract total_frames
                total_frames = (int(video.fps * video.duration))

                # Initialize output matrix for video
                fps = video.fps
                objects_in_vid = np.zeros((21,total_frames/fps), dtype=int)

                # Start time keeper
                pbar = tqdm(total=total_frames, desc=movie)

                for idx, frame in enumerate(video.iter_frames(fps=video.fps, dtype='uint8')):
                    if idx % fps == 0: 
                        input_tensor = preprocess_image(frame)
                        fcn_output = fcn(input_tensor)['out']
                        output_matrix = torch.argmax(fcn_output.squeeze(), dim=0).detach().cpu().numpy()
                        objects_in_image = np.unique(output_matrix)
                        for object in objects_in_image:
                            if object != 0:
                                objects_in_vid[object,idx] == 1
                    pbar.update(1)

                pbar.close()
                np.savez(savepath + movienoextension + '.npz', movie=objects_in_vid)
        
                # Close the video file
                video.reader.close()
            else:
                print(f"{movienoextension} rgb movie already exists!")
    
    #return list of saved movie .npz instead

# 0=background
# 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
# 6=bus, 7=car, 8=cat, 9=chair, 10=cow
# 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

# yt
# Define the helper function
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0), # 0=background
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb