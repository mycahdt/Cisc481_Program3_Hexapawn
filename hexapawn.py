import numpy as np


# initialState is a vector of 10 values that represents a state in Hexapawn
# 1st value is 0 or 1 which specifies whose turn it is
# Other 9 values specifies the board's state in row major order
# in which 0 is an empty square, and 1 is a white pawn, and -1 is a black pawn
list1 = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]
initialState = np.array(list1)
print(initialState)

list2 = [0, -1, 0, -1, 0, -1, 0, 1, 1, 1]
board2 = np.array(list2)

list3 = [1, -1, 0, -1, 0, -1, 0, 1, 1, 1]
board3 = np.array(list3)

class Game:
    def __init__(self, state):
        # Sets the attributes of the Game class
        self.player = self.ToMove(state)
        self.board = self.getBoard(state)
        self.state = state
        
    # Helper function: getBoard(state) that takes in a state, and
    # returns the state as a 2D array
    def getBoard(self, state):
        board = [[state[1], state[2], state[3]],
                [state[4], state[5], state[6]],
                [state[7], state[8], state[9]]]
        return board

    # ToMove(state) is a function that takes in a state as input, and
    # returns the player that has the move in a state
    def ToMove(self, state):
        return state[0]

    # Action(state) is a function that takes in a state as input, and
    # returns the list of legal moves in a state
    def Actions(self, state):

        # Initalizes the list of actions
        actions = []

        # Determines the current player
        player = self.ToMove(state)

        # Calls the getBoard() helper function to get the state as a 2D array
        board = self.getBoard(state)

        # Handles if the current player is MAX (white)
        if player == 0:
            for row in range(3):
                for col in range(3):
                    location = str(row) + " " + str(col)
                    if board[row][col] == 1:
                        if row != 0:
                            if board[row-1][col] == 0:
                                advanceAction = "advance " + location
                                actions.append(advanceAction)
                            if col != 2 and board[row-1][col+1] == -1:
                                captureRight = "capture-right " + location
                                actions.append(captureRight)
                            if col != 0 and board[row-1][col-1] == -1:
                                captureLeft = "capture-left " + location
                                actions.append(captureLeft)

        # Handles if the current player is MIN (black)
        else:
            for row in range(3):
                for col in range(3):
                    location = str(row) + " " + str(col)
                    if board[row][col] == -1:
                        if row != 2:
                            if board[row+1][col] == 0:
                                advanceAction = "advance " + location
                                actions.append(advanceAction)
                            if col != 2 and board[row+1][col+1] == 1:
                                captureRight = "capture-right " + location
                                actions.append(captureRight)
                            if col != 0 and board[row+1][col-1] == 1:
                                captureLeft = "capture-left " + location
                                actions.append(captureLeft)

        # Returns the list of legal moves for the given state
        return actions


    # Result(state, actions) is a function that takes in a state an action,
    # and returns the resulting state of that action 
    def Result(self, state, action):

        # Gets the location of the pawn to move
        myAction = action.split(' ')
        myRow = int(myAction[1])
        myCol = int(myAction[2])

        # Initializes the new state
        myStateList = []

        # Determines the current player
        player = self.ToMove(state)

        # Adds the next player to the new state
        if player == 1:   
            myStateList.append(0)
        else:
            myStateList.append(1)

        # Calls the getBoard() helper function to get the state as a 2D array
        board = self.getBoard(state)

        # Sets the current player's position to an empty spot
        board[myRow][myCol] = 0

        # Handles if the current player is MAX (white)
        if player == 0:

            # Handles an advance move
            if myAction[0] == 'advance':
                board[myRow-1][myCol] = 1
            # Handles a capture-left move
            elif myAction[0] == 'capture-left':
                board[myRow-1][myCol-1] = 1
            # Handles a capture-right move
            else:
                board[myRow-1][myCol+1] = 1

        # Handles if the current player is MIN (black)
        else:

            # Handles an advance move
            if myAction[0] == 'advance':
                board[myRow+1][myCol] = -1
            # Handles a capture-left move
            elif myAction[0] == 'capture-left':
                board[myRow+1][myCol-1] = -1
            # Handles a capture-right move
            else:
                board[myRow+1][myCol+1] = -1

        # Adds the updated values of the board to the state list
        for row in range(3):
            for col in range(3): 
                myStateList.append(board[row][col])


        # Sets the state into a vector
        resultState = np.array(myStateList)


        return resultState
        

    # IsTerminal(state) is a function that takes in a state as input, and
    # returns true when the Game is over, and false otherwise.
    # The Game is over when a player gets one of their pawns 
    # to the other end of the board, or if they make it such that 
    # their opponent is stuck on their next move.
    def IsTerminal(self, state):
        for i in range(10):
            if (i == 1 or i==2 or i==3) and state[i] == 1:
                return True
            if (i == 7 or i==8 or i==9) and state[i] == -1:
                return True
            myActions = self.Actions(state)
            if len(myActions) == 0:
                return True
        return False


    # Utility(state) function takes in a state as input, and 
    # returns 0 if the state is not terminal, or 
    # returns 1 is the state is terminal and white wins, or
    # returns -1 is the state is terminal and black wins 
    def Utility(self, state):

        if not self.IsTerminal(state):
            return 0
        
        if self.IsTerminal(state):

            # Returns 1 if White wins, Returns -1 if Black Wins
            return self.ToMove(state)
        
    # Hash function used as the key for the Minimax policy dictionary 
    def hash(self, state):
        board = self.getBoard(state)
        myHash = ''
        for row in range(3):
            for col in range(3):
                if board[row][col] != 0:
                    myHash = myHash + str(board[row][col])
                else:
                    myHash = myHash + ' '
        return myHash

            
        
# Class Minimax that had a policy, based on the game
class Minimax:
    def __init__(self):
        self.policy = {}

    # Function minimaxSearch that takes in a game and a state (vector)
    # returns the move, after calling the macValue() function 
    def minimaxSearch(self, Game, state):
        Game.player = Game.ToMove(state)
        value, move = self.maxValue(Game, state)
        return move

    # Function maxValue that takes in a Game and state
    def maxValue(self, Game, state):

        # Calls the isTerminal function, given the state
        if Game.IsTerminal(state):
            # Returns the result of calling the Utility() function 
            return Game.Utility(state), []
        
        # Sets v to an impossible value 
        v = -2

        # Loops through each action and calls the minValue() function 
        # and adds the action to list of moves  
        move = []      
        for action in Game.Actions(state):
            v2, a2 = self.minValue(Game, Game.Result(state, action))
            if v2 > v:
                v = v2
                move.append(action)
            elif v2 == v:
                move.append(action)

        # Sets the policy, with the key being set with the hash function
        # and the value set given v and the move
        self.policy[Game.hash(state)] = (v, move)
        return v, move

    
    # Function minValue that takes in a Game and state
    def minValue(self, Game, state):

        # Calls the isTerminal function, given the state
        if Game.IsTerminal(state):
            # Returns the result of calling the Utility() function
            return Game.Utility(state), []
        
        # Sets v to an impossible value 
        v = 2

        # Loops through each action and calls the maxValue() function 
        # and adds the action to list of moves
        move = []
        for action in Game.Actions(state):
            v2, a2 = self.maxValue(Game, Game.Result(state, action))
            if v2 < v:
                v = v2
                move.append(action)
            elif v2 == v:
                move.append(action)

        # Sets the policy, with the key being set with the hash function
        # and the value set given v and the move
        self.policy[Game.hash(state)] = (v, move)
        return v, move



#Activiation function: sigmoid()
def sigmoid(x, derivativeBool):
    if(derivativeBool):
        return (1/(1+np.exp(-x)))*(1-(1/(1-np.exp(-x))))
    else:
        return 1/(1+np.exp(-x))


#Activiation function: relu()
def relu(x, derivativeBool):
    if(derivativeBool):
        if x<0:
            return 0
        else:
            return 1
    else:
        return max(0, x)


# Class layer that has a matrix of inputs, a matrix of weights, and a matrix of biasses
class Layer():
    def __init__(self, neurons, inputs):
        
        # Inputs is a given matrix
        self.inputs = inputs
        

        # Creates the matrix of weights, 
        # given the inputs and neurons matrices
        weightsList = []
        for i in range(len(inputs)):
            newList = []
            for n in range(len(neurons)):
                newList.append(np.random.uniform(-1, 1))
            weightsList.append(newList)
        self.weights = np.array(weightsList)


        # Creates the matrix of biases,
        # given the matric of neurons
        biasesList = []
        for n in range(len(neurons)):
            biasesList.append(np.random.uniform(-1, 1))
        biasMatrix = np.array(biasesList)
        self.biases = biasMatrix.transpose()







# Class Network that has layers and an activation function 
class Network:
    def __init__(self, layers, activationFn):
        self.layers = layers
        self.activate = activationFn

    # classify(slef, inputs) function takes in
    # an instance of the network and a vector of inputs
    def classify(self, inputs):

        # Initiallizes the final list
        finalList = []

        # Loops through each layer of the network
        for layer in self.layers:
            # Transposes the input matrix in order to have a vertical vector
            inputMatrix = self.inputs.transpose()

            # Multiplies weights*inputs matrices
            multMatrix = np.dot(layer.weights, inputs)

            # Adds togther the result of (weights*inputs) + bias matrices
            addedMatrix = np.add(multMatrix, layer.biases)

            # Calls activation fucntion and adds it to the final list
            finalList.append(self.activate(addedMatrix))

        # Converts list into a matrix
        finalMatrix = np.array(finalList)

        return finalMatrix
    

    # Funciton updatedWeights takes in the exptected weights and the actual weights 
    # and it updates the wights 
    def updateWeights(self, expectedWeights, actualWeights):
        for layer in range(len(self.layers) - 1, 0, 1):
            if layer == len(self.layers) - 1:
                error = 2 * (expectedWeights - actualWeights)
            else:
                error = np.matmul(change[layer+1], self.layers[layer+1].wights)
            change = error*self.activate(np.array(self.layers[layer].output), der=True)
            self.layers[layer].weights -= change[layer]

        

# Class HexapawnNetwork
class HexapawnNetwork:
    
    # VectorState() function that takes in a state and if it is a state
    def VectorState(self, state, isState):
        vector = []

        # If it is a state, then adds either 
        # 1 for white or -1 for black onto the vector
        if isState:
            if state[0] == 1:
                vector.append(1)
            else:
                vector.append(-1)
            for i in state[2:]:
                if i == 1:
                    vector.append(1)
                elif i == -1:
                    vector.append(-1)
                else:
                    vector.append(0)
        # If it is not a state then the vector is set to 1 
        else:
            vector = [0,0,0,0,0,0,0,0,0]
            for move in range(len(state)):
                count = 0
                for i in range(3):
                    for j in range(3):
                        if state[move][1] == i and state[move][2] == j:
                            vector[count] = 1
                        count += 1
        return vector

    # Takes in a policyTable and the activation function 
    def __init__(self, policyTable, activateFn):

        # Sets the network 
        self.network = Network([Layer(10,10),Layer(10,10)],activateFn)

        # Sets the inVect to input and the expectVext to expected
        input = []
        expected = []
        for key, value in policyTable.items():
            input.append(self.VectorState(key, True))
            expected.append(self.VectorState(value[1], False))
        self.inVect = input
        self.expectVect = expected



def main():
    
    myGame = Game(initialState)
    myMinimax = Minimax()
    #print(myMinimax.minimaxSearch(myGame, initialState))
    myMinimax.minimaxSearch(myGame, initialState)
    print(myMinimax.policy.values())



if __name__=='__main__':
    main()