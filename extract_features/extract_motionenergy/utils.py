''' This script has functions that together extract motion energy features from a movie (you can think about this as extracting moving edges from a movie, but see here and here for more details). This script is more-so a wrapper for pymoten, which is a package for extracting motion energy using a gabor filter pyramid (see here).

This script requires a .json file that has information about the movie, path locations, and the gbor pyramid you want to push the movie through.'''

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
import moten
import PIL.Image as pil_image
from joblib import Parallel, delayed
from time import time
from scipy.signal import convolve
from scipy import signal
import _pickle as pickle



def rgb_to_gray_luminosity(rgb_image):
    start = time()
    ''' Converts RGB image to gray image.
    
    Parameters
    ----------
    rgb_image : RGB image array
    
    Returns
    -------
    gray_image : converted gray array
    
    '''
    red_weight = 0.2989
    green_weight = 0.5870
    blue_weight = 0.1140
    
    gray_image = np.dot(rgb_image[..., :3], [red_weight, green_weight, blue_weight])
    #print timer
    end = time()
    #print("rgb2gray takes " + str(end - start))
    return gray_image.astype(np.uint8)

def load_resize_image(img, hdim, vdim):
    start = time()

    ''' Resizes image to passed hdim and vdim.
    
    Parameters
    ----------
    img : array to be resized
    hdim : horizontal dimension of new array
    vdim : vertical dimensino of new array
    
    Returns
    -------
    image : resized array
    '''
    img = resize(img, (hdim, vdim), anti_aliasing=True)
    end = time()
    #print("load_resize takes " + str(end - start))
    start = time()
    image = img_as_ubyte(img)
    end = time()
    #print("ubyte takes " + str(end - start))
    return image

def movie_to_gray_array(json_filepath):
    ''' Converts movie to gray-scale matrix, and saves it out as an .npz. 
    
    Parameters
    ----------
    json_filepath : the .json file that houses all information to be passed to function
    
    Returns
    -------
    gray_image_data : a gray-scale movie matrix
    
    '''
    
    # Open feature json file
    with open(json_filepath) as f:
        data = json.load(f)

    # Isolate all info for each stimuli
    for stimuli, stimuli_data in data.items():
        hdim = stimuli_data['hdim']
        vdim = stimuli_data['vdim']
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
                gray_image_data = np.empty((hdim, vdim, total_frames), dtype=np.uint8)

                # Start time keeper
                pbar = tqdm(total=total_frames, desc=movie)

                for idx, frame in enumerate(video.iter_frames(fps=video.fps, dtype='uint8')):
                    img = load_resize_image(frame, hdim, vdim)
                    gray_image_data[:, :, idx] = rgb_to_gray_luminosity(img)
                    pbar.update(1)

                pbar.close()
                np.savez(savepath + movienoextension + '_gray.npz', movie=gray_image_data)

                # Close the video file
                video.reader.close()
            else:
                print(f"{movienoextension} gray movie already exists!")
                
    #return list of saved movie .npz instead

def hanning_filter_3D(frames, downsample_factor):
    # Number of frames and frame size.
    num_frames, height, width = frames.shape

    # Create a Hanning window as the low-pass filter.
    hanning_window = np.hanning(downsample_factor)

    # Initialize an array to store the filtered frames.
    filtered_frames = np.zeros((num_frames // downsample_factor, height, width))

    # Apply Hanning filter to each group of 'downsample_factor' frames.
    for i in range(0, num_frames, downsample_factor):
        group_of_frames = frames[i:i+downsample_factor]
        filtered_group = convolve(group_of_frames, hanning_window[:, None, None], mode='same')
        filtered_frames[i // downsample_factor] = np.mean(filtered_group, axis=0)

    #np.savez(savepath + movienoextension + "_downsampledfeatures.npz", features=filtered_frames)
    print("did hanning window! " + str(filtered_frames.shape))
    return filtered_frames
      
def downsample_matrix(matrix, factor):
    original_rows, original_cols = matrix.shape
    new_cols = original_cols // factor

    downsampled_matrix = np.empty((original_rows, new_cols))

    for i in range(original_rows):
        for j in range(new_cols):
            start_idx = j * factor
            end_idx = start_idx + factor
            segment = matrix[i, start_idx:end_idx]
            downsampled_matrix[i, j] = np.sum(segment * signal.windows.hann(factor)) / factor

    return downsampled_matrix

def push_thru_pyramid(json_filepath):
    ''' Pushes gray-scale movie matrix through gabor pyramid. 
        Essentially a wrapper for Pymoten.
        Downsamples to sampling rate of TR, designated in .json.
        Saves out feature .npz file.
    
    Parameters
    ----------
    json_filepath : the .json file that houses all information to be passed to function
    
    '''
    #open feature json file
    with open(json_filepath) as f:
        data = json.load(f)

    # load in json data
    for stimuli, stimuli_data in data.items():
            hdim = stimuli_data['hdim']
            vdim = stimuli_data['vdim']
            fps = stimuli_data['fps']
            tf = stimuli_data['tf']
            sf = stimuli_data['sf']
            sr = stimuli_data['samplerate']
            dirr = stimuli_data['dir']
            downloadpath = stimuli_data['downloadpath']
            movies = stimuli_data['movies']
            savepath = stimuli_data['savepath']
            
            # create pyramid object, using parameters outlined in .json file
            pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(hdim, vdim), 
                                       stimulus_fps=fps, 
                                       temporal_frequencies=tf, 
                                       spatial_frequencies=sf, 
                                       spatial_directions=dirr)
            
            filename = savepath + "pyramid.obj"
            filehandler = open(filename, 'wb') 
            pickle.dump(pyramid, filehandler)
            print("saved pyramid!")
            filterdictionary = pyramid.filters
            np.savez(savepath + "filters.npz", filters = filterdictionary)
            print("saved filters!")
            
            # for each movie, load in data and extract motion energy from movie
            for movie in movies:
                moviepath = os.path.join(downloadpath, movie)
                movienoextension = movie[:len(movie)-4]
                
                if not os.path.exists(savepath + movienoextension + "_downsampledfeatures.npz"):
                    data = np.load(savepath + movienoextension + "_gray.npz", allow_pickle=True)['movie']
                    #data expected to be nimages, vdim, hdim
                    reorganized_array = np.transpose(data, (2, 0, 1))
                    features = pyramid.project_stimulus(reorganized_array)
                    np.savez(savepath + movienoextension + "_features.npz", features=features.T)
                    # Example usage
                    print(features.shape)
                    downsample_factor = sr*fps
                    downsampled_matrix = downsample_matrix(features.T, downsample_factor)
                    np.savez(savepath + movienoextension + "_downsampledfeatures.npz", features=downsampled_matrix)
                else:
                    print(f"down_sampled features for {movie} already done!")
    return None
    
def save_cleaned_features(json_filepath):
    with open(json_filepath) as f:
        data = json.load(f)

    # load in json data
    for stimuli, stimuli_data in data.items():
            fps = stimuli_data['fps']
            sr = stimuli_data['samplerate']
            dirr = stimuli_data['dir']
            downloadpath = stimuli_data['downloadpath']
            movies = stimuli_data['movies']
            savepath = stimuli_data['savepath']
            TRs = stimuli_data['TRs']

            x_train = []
            x_test = []

            ### need to edit this to take in correct movies
            for movie,runs in TRs.items():
                features = np.load(f"/home/jovyan/workingdirectory/{movie}_downsampledfeatures.npz", allow_pickle=True)['features']
                for key, run in runs.items():
                    if key.startswith('train'):
                        x_train.append(features[:,run[0]:run[1]])
                    if key.startswith('test'):
                        x_test.append(features[:,run[0]:run[1]])

            X_train = np.concatenate(x_train, axis=1)
            X_test = np.stack(x_test)
            np.savez(savepath + "me_features_all.npz", x_train=X_train, x_test=X_test)
            
    return None

                
    
    
