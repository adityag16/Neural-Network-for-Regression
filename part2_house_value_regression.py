import torch
import torch.nn as nn
import torch.optim as opt
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import csv
import random
from sklearn.model_selection import train_test_split



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_neurons, output_dim=1):
        super(NeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.num_hidden_neurons = num_hidden_neurons
        self.layers = []
        self.output_dim = output_dim

        for i in range(len(num_hidden_neurons)+1):
            if(i == 0):
                self.layers.append(nn.Linear(input_dim, num_hidden_neurons[i]))
                self.layers.append(nn.ReLU())
            elif(i != len(num_hidden_neurons)):
                self.layers.append(nn.Linear(num_hidden_neurons[i-1], num_hidden_neurons[i]))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(num_hidden_neurons[i-1], output_dim))

        self.net = nn.Sequential(*self.layers)

        
    def forward(self, _input):
        
        _input = self.net(_input)

        return _input


class Regressor():

    def __init__(self, x, nb_epoch, learning_rate, num_hidden_neurons, batch_size):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_function = nn.MSELoss()
        self.num_hidden_neurons = num_hidden_neurons

        
        self.preprocessing_min = 0
        self.preprocessing_max = 0
        self.preprocessing_avg = 0
        self.NeuralNetwork = NeuralNetwork(self.input_size, self.num_hidden_neurons)

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """
        
        
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own

        onehot_encoded = pd.get_dummies(x['ocean_proximity'], prefix="ocean_proximity")
        x = pd.concat([x, onehot_encoded], axis = 1)
        x.drop(columns = 'ocean_proximity', axis = 1, inplace = True)

        
        if(training == True):
            self.preprocessing_avg = x.mean()
            self.preprocessing_min = x.min()
            self.preprocessing_max = x.max()
        
        x.fillna(self.preprocessing_avg, inplace = True)

        x_normalised = (x-self.preprocessing_min)/(self.preprocessing_max-self.preprocessing_min)
        # Return preprocessed x and y, return None for y if it was None
        return torch.tensor(x_normalised.to_numpy()), (torch.tensor(y.to_numpy()) if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        optimiser = opt.Adam(self.NeuralNetwork.parameters(), lr=self.learning_rate)
        for _ in range (self.nb_epoch):
        
            indices = np.random.permutation(len(X))
            shuffled_inputs = X[indices]
            shuffled_outputs = Y[indices]

            batch_inputs = torch.split(shuffled_inputs, self.batch_size)
            batch_outputs = torch.split(shuffled_outputs, self.batch_size)


            for i in range(len(batch_inputs)): #iterate over batches

                predicted_output = self.NeuralNetwork(batch_inputs[i].float())
                loss = self.loss_function(predicted_output, batch_outputs[i].float())

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        predicted_output = self.NeuralNetwork(X.float())

        return predicted_output.detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        
        
        predicted_output = self.predict(x)
    
        return mse(Y.detach().numpy(), predicted_output, squared=False)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x_train, y_train, x_val, y_val): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    fields = ["Learning Rate", "Epochs", "Hidden Layer Neurons", "Batch Size", "RMSE", ]
    best_rmse = np.inf
    best_params = {}
    with open('data.csv', mode='w') as data:
        writer = csv.writer(data)
        writer.writerow(fields)
        for _ in range(1):
            epochs = random.randint(300,1000)
            batch_size = random.randint(250,500)
            neuron_no = random.sample(range(10,40), random.randint(3,7))
            learning_rate = random.uniform(0.07,0.1)
            
            tuning_model = Regressor(x_train, nb_epoch= epochs, learning_rate= learning_rate, num_hidden_neurons= neuron_no , batch_size= batch_size)
            tuning_model.fit(x_train,y_train)
        
            RMSE = tuning_model.score(x_val,y_val)

            param_values = [learning_rate, epochs, neuron_no, batch_size, RMSE]
            
            writer.writerow(param_values)


            if(RMSE < best_rmse):
                best_rmse = RMSE
                save_regressor(tuning_model)
                best_params = {"epochs":epochs, "batch_size":batch_size, "neuron_no": neuron_no, "learning rate": learning_rate, "RMSE":RMSE}

    return  best_params


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
 

    regressor = Regressor(x_train, nb_epoch = 765, learning_rate=0.09002852101468667, num_hidden_neurons=[31, 28, 32, 21], batch_size=410)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    # error = regressor.score(x_train, y_train)
    # print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()


        




