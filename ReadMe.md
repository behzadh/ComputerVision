# Computer Vision Rock Paper Scissors Project Documentation Guideline

> Rock Paper Scissors Game is a game that asks two users for a hand sign of Rock, Paper or Scissors in the same time and defines a winer based on some conditions. 

- In this project we design the game to randomly choose an option (rock, paper, or scissors) and then ask the user for an input.

- User's input will first be entered manualy and without the camera. Afterwards, we use camera to get the input of the user via the webcam.

- To start, we need to install all libraries and python packages that we need for this project.

## Milestone 1

> In order to disign the code, we need to install all libraries and dependencies.

- Create a virtual environment:

    - First, create a conda environment and then install the necessary requirements as well as opencv-python, tensorflow, and ipykernel.

    - After creating the environment, activate it, and then install pip as following:
    ```code
    conda install pip
    ```
    
    Now in order to install any libraries, you can run:
    ```code
    pip install <library>
    ```

    - Important: If you are on Ubuntu, the latest version of Tensorflow doens't work with the default version of Python. When creating the virtual environment, use Python 3.8 instead by running:
    ```code
    conda create -n my_env python=3.8
    ```

    Where my_env is the name of the environment you want to create.

    - Important! If you are using a Mac M1 chip, this task will be a bit longer. Once you installed miniconda. First, create a virtual environment by running the following commands:
    ```code
    conda create -n tensorflow-env python=3.9

    conda activate tensorflow-env

    conda install pip
    ```

    Then, follow the steps from the section that says "arm64: Apple Silicon" from this [link](https://developer.apple.com/metal/tensorflow-plugin/).

    - Once you get tensorflow for Mac, you will install opencv for Mac and ipykernel by running the following commands:
    ```code
    conda install -c conda-forge opencv

    pip install ipykernel
    ```
- Now, we have everything to start...

## Milestone 2

> Play the game without camera:

- The code is written within a class in "manual_rps.py" and has the following functions:

    - Init function to initializing all attributes
    - get_computer_choice to randomly choose an option for computer
    ```python
    def get_computer_choice(self):
        ## This functions randomly picks a choice for computer
        return random.choice(self.computer_selections).lower()
    ```
    - get_user_choice which is the input function to get the user's choice
    ```python
    def get_user_choice(self):
        ## This functions asks the user for a choice to start the game
        user_input = input('Please pick one of Rock Paper or Scissor -> ').lower()
        while user_input not in ['rock', 'paper', 'scissor']:
            user_input = input('Your choice should be one of Rock Paper or Scissor -> ')
        return user_input.lower()
    ```
    - get_winner will define the conditions based on Rock, Paper or Scissors game to get the winner.
    - And the play function that defines the rule of the play.

- In order to play the game you need to run the code in a terminal as following:
```code
python manual_rps.py
```
The game will starts and automatically get a choice for the computer and will ask the user:
```code
Please pick one of Rock Paper or Scissor -> 
```
Base on the two choices the winner will be announced.

## Milestone 3

> In order to play the game with camera we need to do two modifications:
    1. We need to train computer to predict the user's sign.
    2. We also need to change the get_user_choice function to a prediction function to replace the manual input for the output of the computer vision model.

### Train the Computer:

- To start we train computer to recognize what are the rock, paper, scissors and nothing signs of the user. This can be done by using the following website to train four classes of Rock Paper Scissor or None:
```code
https://teachablemachine.withgoogle.com/
```
- Once the training is done we download the keras model to our work directory. Please download the model from the "Tensorflow" tab in Teachable-Machine. The model should be named 'keras_model.h5' and the text file containing the labels should be named 'labels.txt'.

- Run the code below just to check the model you downloaded is working as expected:
```python
import cv2
from keras.models import load_model
import numpy as np
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True: 
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    cv2.imshow('frame', frame)
    # Press q to close the window
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
```
Please make sure that you have the correct name for the model, the correct name for the labels, and that the model and this file are in the same folder.

### Predict the output of the computer vision model:

> We create a new file called "camera_rps.py" which will be very simillar to previous "manual_rps.py" and create a new function called get_prediction that will return the output of the keras model.

- To load the keras model to our code we do:
```python
from keras.models import load_model

model = load_model('keras_model.h5', compile=False)
```

- As model is loaded, we can start geting frames from the webcam and predict the output of the computer vision model.

```python
ret, frame = self.cap.read()
    if ret==True:
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 255.0) #- 1 # Normalize the image
        data[0] = normalized_image
        prediction = model.predict(data)
```

The result is a list of probabilities for each class and it picks the class with the highest probability. 

- In reality, when you play a regular game, you usually count down to zero, and at that point you show your hand. In order to simulate this, a countdpown has been added to the code and when the countdown gets to zero, the camera will capture the input.

- The code will print the countdown and all other inforamation (like results) in the webcam display.

<img src="https://github.com/behzadh/ComputerVision/blob/main/images/count1.png" width="300"> <img src="https://github.com/behzadh/ComputerVision/blob/main/images/choices.png" width="300"> <img src="https://github.com/behzadh/ComputerVision/blob/main/images/results.png" width="300">

## Conclution

> In this project we designed the Rock Paper Scissors game to be played both manually and via webcam.

- In order to play the game, you need to do the following steps:
    1. Go to this [website](https://teachablemachine.withgoogle.com/) and click on Image Project to get your model.
    2. You need to provide four classes with this order: Nothing, Rock, Paper and Scissor. 
    3. Train your model and download it. Note your label.txt file should be like:
    ```code
    0 Nothing
    1 Rock
    2 Paper
    3 Scissor
    ```
    4. Copy your model () and label () files to your work directory.
    5. Run the foloowing command to start the game:
    ```code
    python camera_rps.py
    ```
