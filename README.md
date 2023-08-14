# visual-feature-decoding
Neurohack 2023: develop a feature extraction package and visual feature analyses for visual stimuli data


- The first goal is to begin by making a classifier script. We will use 7T NSD fMRI data collected whilst people viewed scene images and use documented stimulus labels and pre-computed models to decode visual representations of seen images across the brain. 
- The second, parallel goal is to create a package that extracts visual features from visual stimuli. We will work with the HCP 7T movies and data to do this. A nice description of the movies can be found in Finn and Bandettini, 2021.
  Visual features
  Semantic labels (CNNs) 
  Low-level, motion energy pyramid (Could use pymoten)
  Texture features (Henderson 2023, also has extraction for semantics & gabors)
After extracting features from neural data:
  Train encoding models
  Train decoding classifiers
- (if time permits) The final goal is to compare the outcomes of the models in steps 1 and 2 to the neural data to see whether there are comparable neural responses across datasets when using similar features/labels
