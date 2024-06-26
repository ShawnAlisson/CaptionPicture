# Image Captioning API

This is an image captioning API built with Flask, allowing users to upload images and generate captions using various deep learning models.

## Features

- Supports multiple deep learning models for image captioning, including MobileNetV2, InceptionV3, ResNet50, Transformer and BLIP.
- Users can upload images through a simple web interface.
- Provides real-time captioning with processing time displayed.

## Prerequisites

Before running the application, make sure you have the following installed:

- Python 3.x
- pip (Python package manager)

## Installation

1. Clone the repository:
```sh
git clone https://github.com/ShawnAlisson/CaptionPicture.git
```

2. Navigate to the project directory:
```sh
cd CaptionPicture
```
3. Install the required Python packages:
```sh
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```sh
python app.py
```
2. Open a web browser and go to `http://localhost:8000` to access the web interface.

## Models

- **MobileNetV2**: A lightweight convolutional neural network architecture.
- **InceptionV3**: A deep convolutional neural network architecture.
- **ResNet50**: A residual neural network architecture.
- **Transformer**: A transformer-based model for image captioning.
- **BLIP**: A state-of-the-art vision-language transformer model.
