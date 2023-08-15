## General idea of how to design scripts (edit if you update the script format :-) )

1. Take in movie
2. Based on package used to extract features, have code that converts to appropriate input (.mp4, array, png).
3. Push that stimuli in the correct format through the extractor package (or extractor algorithm)
4. Save features in .npz or .hdf file

## General folder directory to expect

WORKINGDIRECTORY  
  |__ movie.mp4             --> movie to be converted
  |__ pngs                  --> folder for pngs if the script needs to convert from movie to png
      |__ png.001
      |__ png.002
  |__ .npz or .ndf file     --> output of script, features.npz or features.hdf
  |__ intermediate/tmp      --> folder for intermediate or temp files