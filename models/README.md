# Models

This subfolder contains various scripts related to model architecture, training, and feature extraction. The models were selected to include a range of different methods as building blocks. For models with specific loss functions, we take only the backbone.

## Scripts

1. [cornet_s.py](cornet_s.py): Script for Cornet-S model architecture.
   - **Paper:** [Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs](https://arxiv.org/abs/1909.06161)

2. [extract_model_activations.py](extract_model_activations.py): Script for extracting model weights after passing stimuli.

3. [FaceNet.py](FaceNet.py): Script for FaceNet model architecture.
   - **Paper:** [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

4. [inception.py](inception.py): Script for Inception model architecture.
   - **Paper:** [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

5. [mobilenet.py](mobilenet.py): Script for MobileNet model architecture.
   - **Paper:** [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

6. [models_train.py](models_train.py): This script is used for training different models. You can specify the model architecture and training parameters within the script.

7. [resnet.py](resnet.py): Script for ResNet model architecture.
   - **Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

8. [SphereFace.py](SphereFace.py): Script for SphereFace model architecture.
   - **Paper:** [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

9. [vgg.py](vgg.py): Script for VGG model architecture.
   - **Paper:** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

## Usage

### Training Script

Run the training script using the command: `python models_train.py --dataset <data you Want to use> --analysis_type <type of analysis> --model <the model you want to train> --batch_size 256 --lr 0.1 --optimizer "sgd"`

### Activations Script

Run the activation extraction script using the command: `python extract_model_activations.py --n_input_channels 1 --model <the model you want to train> --analysis_type <type of analysis> --in_weights <path to model weights>`

