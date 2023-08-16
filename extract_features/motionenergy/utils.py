''' This script has functions that together extract motion energy features from a movie (you can think about this as extracting moving edges from a movie, but see here and here for more details). This script is more-so a wrapper for pymoten, which is a package for extracting motion energy using a gabor filter pyramid (see here).

This script requires a .json file that has information about the movie, path locations, and the gbor pyramid you want to push the movie through.'''

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import subprocess
from skimage import io
from skimage.transform import resize
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


def rgb_to_gray_luminosity(rgb_image):
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
    return gray_image.astype(np.uint8)

def load_resize_image(img, hdim, vdim):
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
    img = resize(img, (hdim, vdim))
    image = img_as_ubyte(img)
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
            output_directory = f'{savepath} + {movienoextension}/'
            
            # If the output directory for the movie doesn't exist, make it
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                
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
            
    #return list of saved movie .npz instead

def push_thru_pyramid(json_filepath):
    ''' Pushes gray-scale movie matrix through gabor pyramid. 
        Essentially a wrapper for Pymoten.
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
            dir = stimuli_data['dir']
            downloadpath = stimuli_data['downloadpath']
            movies = stimuli_data['movies']
            savepath = stimuli_data['savepath']
            
            # create pyramid object, using parameters outlined in .json file
            pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(hdim, vdim), 
                                       stimulus_fps=fps, 
                                       temporal_frequencies=tf, 
                                       spatial_frequencies=sf, 
                                       spatial_directions=dir)
            
            # for each movie, load in data and extract motion energy from movie
            for movie in movies:
                movienoextension = movie[:len(movie)-4]
                data = np.load(savepath + movienoextension + "_gray.npz", allow_pickle=True)['movie']
                #data expected to be nimages, vdim, hdim
                reorganized_array = np.transpose(data, (2, 0, 1))
                features = pyramid.project_stimulus(reorganized_array)
                np.savez(savepath + movienoextension + "_features.npz", features=features)
                
                
    
    
