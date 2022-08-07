'''
This code is written to design Rock Paper Scissors game. 
This code randomly choose an option (rock, paper, or scissors) and then ask the user for an input to find the winner
@author: Behzad on 5 August 2022
'''
import random

class ComputerVision:
    def __init__(self, num_lives=1, max_score=3):
        ## Initializing all attributes
        self.num_lives = num_lives
        self.max_score = max_score
        self.user_score = 0
        self.computer_score = 0
        self.computer_selections = ['Rock', 'Paper', 'Scissor']

    def get_computer_choice(self):
        ## This functions randomly picks a choice for computer
        return random.choice(self.computer_selections).lower()
    
    def get_user_choice(self):
        ## This functions asks the user for a choice to start the game
        user_input = input('Please pick one of Rock Paper or Scissor -> ').lower()
        while user_input not in ['rock', 'paper', 'scissor']:
            user_input = input('Your choice should be one of Rock Paper or Scissor -> ')
        return user_input.lower()

    def get_winner(self, computer_choice, user_choice):
        ## Running the game until the user wins or loses
        print(f'Computer choice is * {computer_choice} * and your choice is * {user_choice} *')
        if computer_choice == user_choice:
            print('It is a draw! Try it again..')
        elif computer_choice == 'Rock':
            if user_choice == 'Paper':
                self.user_score += 1
                print('Congratulations, you won!')
            else:
                #self.num_lives = self.num_lives - 1
                self.computer_score += 1
                print('Computer won!')
        elif computer_choice == 'Paper':
            if user_choice == 'Scissor':
                self.user_score += 1
                print('Congratulations, you won!')
            else:
                #self.num_lives = self.num_lives - 1
                self.computer_score += 1
                print('Computer won!')
        else:
            if user_choice == 'Rock':
                self.user_score += 1
                print('Congratulations, you won!')
            else:
                #self.num_lives = self.num_lives - 1
                self.computer_score += 1
                print('Computer won!')

    def play(self):
        ## This function defines the rule of the play
        while self.num_lives > 0:
            self.user_score, self.computer_score = 0, 0
            while self.computer_score < self.max_score:
                if self.user_score < self.max_score:
                    com_choice = self.get_computer_choice()
                    usr_choice = self.get_user_choice()
                    self.get_winner(com_choice,usr_choice)
                    if com_choice != usr_choice:
                        print(f'Your score {self.user_score} - {self.computer_score} Computer score')
                else:
                    break
            if self.computer_score == self.max_score:
                self.num_lives = self.num_lives - 1
                print(f'You have {self.num_lives} lives remain')    

def main():
    ## main function 
    game = ComputerVision().play()


if __name__ == '__main__':
    main()