# Images Data Processing

This folder contains a collection of scripts designed to prepare the data for different experiments that we will run across the project. The division into HDF5 and repartition scripts is deliberate, aligning with the distinctive structures of our datasets.

## Datasets

### VGGFace Dataset

- **Dataset:** VGGface
- **Number of Identities:** 2,622
- **Kaggle Dataset:** [VGGface on Kaggle](https://www.kaggle.com/datasets/hearfool/vggface2)
- **Citation:** [Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
  - **Authors:** Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman
  - **Conference:** British Machine Vision Conference
  - **Year:** 2015

### CelebA Dataset

- **Dataset:** CelebA Dataset
- **Number of Identities:** 10,177
- **Number of Face Images:** 202,599
- **Number of Landmark Locations:** 5
- **Number of Binary Attribute Annotations per Image:** 40
- **Dataset Homepage:** [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Kaggle Dataset:** [CelebA Kaggle Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- **Citation:** [Deep Learning Face Attributes in the Wild](https://arxiv.org/abs/1411.7766v3)
  - **Authors:** Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang
  - **Conference:** Proceedings of International Conference on Computer Vision (ICCV)
  - **Year:** 2015

### ImageNet Dataset

- **Dataset:** ImageNet
- **Number of Images:** 14 million
- **Dataset Homepage:** [ImageNet Homepage](https://www.image-net.org/)
- **Kaggle Dataset:** [ImageNet Kaggle Dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)
- **Citation:** [ImageNet: A large-scale hierarchical image database](https://ieeexplore.ieee.org/document/5206848)
  - **Authors:** Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei
  - **Conference:** 2009 IEEE Conference on Computer Vision and Pattern Recognition
  - **Year:** 2009

## Scripts Overview

1. [Repartition VGGFace Data](01-repartition_vggface.py): Repartitions the VGGFace dataset.

   - Assigns classes to each picture based on specified repartition counts (train, valid, test).
   - Creates data repartition folders for training, validation, and testing.
   - Copies pictures to their respective folders based on assigned classes.

2. [Convert VGGFace Data to HDF5](02-vggface_to_h5.py): Converts VGGFace dataset images to HDF5 format.

   - Resizes images to 224x224 pixels.
   - Reads images from the specified data folders (train, valid, test).
   - Converts images to numpy arrays.
   - Stores the image arrays and corresponding labels in HDF5 files.

3. [Convert Stimuli Images to HDF5](03-stimuli_to_h5.py): Converts stimulus images to HDF5 format.

   - Resizes stimulus images to 224x224 pixels.
   - Reads and processes stimulus images from the specified dataset folders.
   - Converts images to numpy arrays.
   - Stores the image arrays in an HDF5 file for efficient storage and retrieval.

4. [Match MEG Subjects to CelebA Identities](04-meg_celeba_id_matching.py): Matches MEG subjects to CelebA dataset identities based on facial recognition.

   - Loads stimulus images from the stimuli directory.
   - Processes stimulus images to extract facial features and create facial embeddings.
   - Compares the facial embeddings of stimulus images with embeddings from the CelebA dataset.
   - Selects CelebA identities that closely match the facial features of the stimuli.
   - Plots the selected images for verification.

5. [Add MEG Stimuli to CelebA Dataset](05-add_meg_stimuli_to_celeba.py): Adds MEG stimulus images to the CelebA dataset.

   - Loads the existing CelebA dataset labels from 'identity_CelebA.txt'.
   - Updates the labels for matched MEG stimulus IDs, ensuring that each matched ID has a maximum of 30 associated images.
   - Copies MEG stimulus images to the 'img_align_celeba' directory, renaming and reindexing them.
   - Appends the updated labels to 'identity_CelebA_with_meg_stimuli.txt'.
   - Saves a mapping dictionary associating MEG stimulus names with their new CelebA IDs.

6. [Create CelebA Dataset with MEG Stimuli](06-create_celeba_dataset_with_stimuli.py): Creates a modified CelebA dataset with MEG stimuli.

   - Loads the original CelebA dataset labels and attributes.
   - Corrects gender attributions in the dataset.
   - Combines the cleaned dataset with MEG stimuli information.
   - Generates new IDs for the dataset to ensure uniqueness.
   - Assigns classes to each picture based on class repartition counts.
   - Creates train, test, and valid text files and repartitions the data.
   - Saves the modified dataset as 'celebA_with_stimuli.csv' and 'celebA_without_stimuli.csv'.
   - Generates class distribution files for both datasets.

7. [Convert CelebA to HDF5](07-celeba_to_h5.py): Converts the CelebA dataset to HDF5 format.

   - Defines functions for calculating facial landmarks, creating square crops based on detection boxes, and transforming images.
   - Utilizes the FaceAlignment library to obtain facial landmarks.
   - Transforms images by cropping them based on facial landmarks and resizing them to a specified size.
   - Processes both "with stimuli" and "without stimuli" datasets for train, test, and valid sets.
   - Stores results in HDF5 format.

8. [Repartition ImageNet Data](08-repartition_imagenet.py): Repartitions the ImageNet dataset to create subsets.

   - Provides functions to retrieve image filenames from directories, duplicate selected data to new locations, and integrate face images into specific directories.
   - Copies a carefully selected subset of data from class directories within the ImageNet training set, creating a refined training subset.
   - Randomly selects and incorporates face images from the CelebA dataset into the new training subset.
   - Filters and selects additional face images specifically for the validation set while updating the validation labels CSV file as needed.

9. [Convert ImageNet to HDF5](09-imagenet_to_h5.py): Converts ImageNet dataset images to HDF5 format.

   - Resizes images to a standard size of 224x224 pixels.
   - Maps analysis types to labels to be dropped during processing.
   - Processes the training and validation sets, excluding specific labels based on the analysis type.
   - Stores results in HDF5 format.

## Files

The "files" subfolder in this directory contains essential data files used by the scripts or where script results are stored. These files include:

These files are referenced and manipulated by the data processing scripts to facilitate data preparation and analysis. Make sure to check this folder for any necessary input files or examine it for the results generated by the scripts.

## Usage

- The scripts are designed to be executed independently within their respective dataset folders. However, when working with a specific dataset, it's advisable to run the scripts sequentially, as each script often builds upon the outputs of the previous ones.
- Depending on the size and complexity of the data, script execution times may vary. Be sure to allocate sufficient computing resources and monitor the progress accordingly.

## Note

This repository showcases an example of how to organize data processing scripts effectively. Customize the scripts and settings to match your specific data and analysis requirements.

Your contributions and suggestions for improvements to this directory are highly encouraged. We hope you find these scripts valuable for your data processing tasks.
