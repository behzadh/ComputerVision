'''
This code is written to design Rock Paper Scissors game. 
This code randomly choose an option (rock, paper, or scissors) and then captures the user's choice by webcam to find the winner
@author: Behzad on 7 August 2022
'''
import random
from traceback import print_tb
import cv2
from keras.models import load_model
import numpy as np
import time

class ComputerVision:
    def __init__(self, num_lives=1, max_score=2, num_counter=5):
        ## Initializing all attributes
        self.num_lives = num_lives # If player wants to play multiple sets of max_score, num_lives should be > 1
        self.max_score = max_score
        self.user_score = 0
        self.computer_score = 0
        self.counter = num_counter # Sets the number of seconds for the count down
        # Setting the computer choices and load the keras model
        self.computer_selections = ['Rock', 'Paper', 'Scissor']
        self.model = load_model('keras_model.h5', compile=False)

    def get_computer_choice(self):
        ## Randomly pick a choice for computer
        return random.choice(self.computer_selections).lower()
    
    def set_text(self, frame, text, x, y, color=(0,0,255), thickness=4, size=1):
        ## Setup to put text on the screen
        if x is not None and y is not None:
            cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

    def text_position(self, text, frame):
        ## Get x, y coordinates of the text on the screen
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        text_X = (frame.shape[1] - textsize[0]) / 2
        text_Y = (frame.shape[0] + textsize[1]) / 2
        return text_X, text_Y

    def draw_text(self, my_string, frame, wait_time=2000, size=1):
        ## Draw and print the text on the screen based on the x and y positions
        mystr = my_string
        print(mystr)
        x_pos, y_pos = self.text_position(mystr, frame)
        self.set_text(frame, mystr, x_pos, y_pos, size=size)
        cv2.imshow("frame", frame)
        cv2.waitKey(wait_time)
    
    def get_prediction(self,comp_choice):
        ## This function captures user's choice via the webcam
        init_time = time.time()
        counter = self.counter
        final_timeout = init_time + counter + 1
        counter_timeout_text = init_time + 1
        counter_timeout = init_time + 1
        while True: 
            ret, frame = self.cap.read()
            if ret==True:
                resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
                image_np = np.array(resized_frame)
                normalized_image = (image_np.astype(np.float32) / 255.0) #- 1 # Normalize the image
                self.data[0] = normalized_image
                # Count down the time and print it om screen
                if (time.time() > counter_timeout_text and time.time() < final_timeout):
                    count_str = 'Count down ' + str(counter)
                    x_pos, y_pos = self.text_position(count_str, frame)
                    self.set_text(frame, count_str, x_pos, y_pos) # Draw counter on the webcam screen
                    counter_timeout_text+=0.03333
                if (time.time() > counter_timeout and time.time() < final_timeout):
                    counter-=1
                    counter_timeout+=1
                cv2.imshow('frame', frame)

                # User's choice will be picked after the counter is zero or by pressing q 
                if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() > final_timeout):
                    prediction = self.model.predict(self.data) # prediction will be calculated based on the trained model
                    max_input = prediction.argmax() # find the position (index) of the most probable prediction
                    # Classifying the four class of nothing = 0, rock = 1, paper = 2 and scissor = 3
                    if max_input == 0:
                        ret, frame = self.cap.read()
                        self.draw_text("Sorry we couldn't recognize your choice, please enter it manually", frame) # Draw it on webcam screen
                        user_input = input('Please enter your choice: Rock Paper or Scissor -> ').lower() # Get the user's choice manually in the terminal
                    elif max_input == 1:
                        user_input = 'rock'
                    elif max_input == 2:
                        user_input = 'paper'
                    else:
                        user_input = 'scissor'
                    break
            else:
                break
        ret, frame = self.cap.read()
        self.draw_text(f"Bot's choice is * {comp_choice} * and yours is * {user_input} *", frame) # Draw Bot and user's choices on the screen
        return user_input.lower()

    def get_winner(self, computer_choice, user_choice):
        ## Define the rules of the game
        if computer_choice == user_choice:
            # If both choices are equal, the game will be repeated
            ret1, frame1 = self.cap.read()
            self.draw_text("It is a draw! Let's try it again..", frame1) # Draw it on the webcam screen
        elif computer_choice == 'rock':
            if user_choice == 'paper':
                self.user_score += 1
                print('Congratulations, you won!')
            else:
                self.computer_score += 1
                print('Computer won!')
        elif computer_choice == 'paper':
            if user_choice == 'scissor':
                self.user_score += 1
                print('Congratulations, you won!')
            else:
                self.computer_score += 1
                print('Computer won!')
        else:
            if user_choice == 'rock':
                self.user_score += 1
                print('Congratulations, you won!')
            else:
                self.computer_score += 1
                print('Computer won!')

    def play(self):
        ## This function runs the play until there will be a winer
        self.cap = cv2.VideoCapture(0) # Starts the webcam
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # Convert images to a numpy array
        while self.num_lives > 0:
            self.user_score, self.computer_score = 0, 0
            while self.computer_score < self.max_score:
                if self.user_score < self.max_score:
                    com_choice = self.get_computer_choice() # Get bot's choice
                    usr_choice = self.get_prediction(com_choice) # Predict user's choice (com_choice added as an arg to be used in a print output, see line 90)
                    self.get_winner(com_choice,usr_choice)
                    if com_choice != usr_choice:
                        ret1, frame1 = self.cap.read()
                        self.draw_text(f'Your score {self.user_score} - {self.computer_score} Computer score', frame1) # Draw scores on the screen
                else:
                    break
            if (self.computer_score == self.max_score) or (self.user_score == self.max_score):
                # If user wants to play moltiple sets, each set will have a winner then one set will be reduced
                self.num_lives = self.num_lives - 1
                if self.num_lives != 0:
                    ret1, frame1 = self.cap.read()
                    self.draw_text(f"This set is fnished, there are {self.num_lives} sets left. Please press 'c' to continue", frame1) # Draw number of remaining sets on the screen
                    if cv2.waitKey(0) & 0xFF == ord('c'):
                        # After each set, user should press c to start a new set
                        pass

            if self.num_lives == 0:
                if self.user_score == self.max_score:
                    ret1, frame1 = self.cap.read()
                    self.draw_text("Congratulations! You won the game :)", frame1, 3000) # Draw result on the screen for 3 seconds
                elif self.computer_score == self.max_score:
                    ret1, frame1 = self.cap.read()
                    self.draw_text("Opss.. You lost! :(", frame1) # Draw result on the screen

        # Release the cap object after the loop 
        self.cap.release()
        # Destroy all windows
        cv2.destroyAllWindows()

def main():
    ## Main function 
    ComputerVision(max_score=3).play()

if __name__ == '__main__':
    main()