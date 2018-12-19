#!/usr/bin/env python
# @Author - David Camacho

import numpy as np
from math import exp
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RHONN:
    '''
    RHONN Implementation
    
    This is an implementation for the general model of the recurrent high order
    neural network (RHONN). Used for the identification of state variables in
    non-linear problems.
    '''
    def __init__(self, net_structure, z_structure=None):
        '''
        Parameters
        ----------
        net_structure - an iterable describing the network structure.
                        e.g. (2, 3, 4) is a network with an input_size=2,
                        output_size=3 (3 states to be identified), and 5
                        elements for each Zi vector.
        z_structure - an iterable describing the structure for each vector Zi
                      and the high order connections.
                      e.g. using the net_structure described before. Where
                      each tuple is a representation for: (input_index, power),
                      and the elements in the array [(1,3), (2,1)] are taken as 
                      a multiplication.
                      [ [ [(1,1)], [(2,2)], [(2,1)], [(1,2)] ],  # Z1
                        [ [(1,2)], [(1,1)], [(2,4)], [(1,3)] ],  # Z2
                        [ [(1,3), (2,1)], [(1,1)], [(1,1)], [(1,1)] ] ] # Z3
        '''
        self.in_size = net_structure[0]
        self.out_size = net_structure[1]
        self.num_z = net_structure[2]
        self.z_structure = z_structure
        self.inline_started = False
        
        # Auxiliar for debug
        self.str_result = 't: {} -- x: {} -- net_out: {} -- error{}'

    def sigmoid(self, x, betha=0.1, alpha=1):
        '''
        Sigmoid fuction used as activation function for the neurons inputs.
        '''
        return alpha*(1/(1+exp(-betha*x)))

    def EKF(self, P, R, H, Q, w, e, eta=1):
        ''' 
        EFK function for the identifier case, for observers is needed a new 
        function.
        '''
        PH = np.matmul(P, H)
        H_TPH = np.matmul(np.matmul(H.T, P), H)
        K = np.dot(PH, np.reciprocal(R + H_TPH))
        
        W_1 = w+np.multiply(np.multiply(eta, K), e)
        
        KH_T = np.matmul(K.reshape(num_z, 1), H.reshape(num_z, 1).T)
        P_1 = P - np.matmul(KH_T, P) + Q
        
        return (W_1, P_1)

    def EKF_weights_adjustment(self, Z, W, errors):
        '''
        Function to calculate EKF weights adjustment, for each state variable.
        '''
        W_1 = np.zeros(W.shape)
        P_1 = np.array([np.identity(self.num_z) for x in range(self.out_size)])

        for i in range(self.out_size):
            W_1[i], P_1[i] = self.EKF(self.EKF_P[i], 
                                      self.EKF_R[i], 
                                      Z[i], 
                                      self.EKF_Q[i], 
                                      W[i], 
                                      errors[i], 
                                      self.EKF_etas[i])

        return W_1, P_1

    def calculate_z(self, s_x):
        '''
        Function to calculate the conections result on each Zi vector.
        '''
        Z = np.ones((self.out_size, self.num_z))
        
        for z in range(self.out_size):
            for l in range(self.num_z):
                for index, power in self.z_structure[z][l]:
                    Z[z, l] *= np.power(s_x[index-1], power)
        return Z
        
    def training_params(self, 
                        activation_funcs=None, 
                        fnc_weights_adjsmnt=None, 
                        EKF_parameters=None, 
                        sigmoid_params=None, 
                        debug=True, 
                        plot_data=True):
        '''
        Function to set the params for the inline/offline training of the RHONN.
        
        Parameters
        ----------
        activation_funcs - an iterable with the activation function for each 
                           neuron input.
        fnc_weights_adjsmnt - a function designed for the weights adjustment of
                              the RHONN, it recives the weights and error from 
                              the time k and returns the k+1 weights.
        EKF_params - an iterable with the parameters for each EKF aplication on
                     the states identification.
                     e.g. using the net_structure metioned before
                     EKF_params = (R, P, Q, etas), where:
                     - R is an iterable with the scalar for each EKF aplication
                       example: [0.1, 0.2, 1.0].
                     - P and Q are diagonal matrices (identity by default)
                     - etas is an iterable with the eta value for each EFK
                       aplication, example: [1, 1, 1].
        sigmoid_params - an iterable of tuples (betha, alpha) with custom values 
                         for each neuron activation.
                         e.g. according the net_structure mentioned before
                         sigmoid_params = [(0.1, 1), # Neuron 1 sigmoid params
                                           (0.6, 1)] # Neuron 2 sigmoid params
        debug - a boolean var to display the result on each iteration of the
                training.
        plot_data - a boolean value for ploting in the result of the 
                    identification and the network error.
        '''
        self.fnc_weights_adjsmnt = fnc_weights_adjsmnt
        self.debug = debug
        self.plot_data = plot_data
        
        # EKF Parameters
        if EKF_parameters is None:
            self.EKF_R = np.random.rand(self.num_z)
            self.EKF_P = [np.identity(self.num_z) for x in range(self.out_size)]
            self.EKF_Q = [np.identity(self.num_z) for x in range(self.out_size)]
            self.EKF_etas = np.ones(self.num_z)
        else:
            self.EKF_R = EKF_parameters[0]
            self.EKF_P = EKF_parameters[1]
            self.EKF_Q = EKF_parameters[2]
            self.EKF_etas = EKF_parameters[3]

        # Sigmoid different funcs
        if activation_funcs is not None:
            self.activation_funcs = activation_funcs
        elif sigmoid_params is not None:
            self.activation_funcs = [partial(self.sigmoid, 
                                             betha=sigmoid_params[x][0], 
                                             alpha=sigmoid_params[x][1]) 
                                     for x in range(len(sigmoid_params))]
        else:
            self.activation_funcs = [partial(self.sigmoid, betha=0.1, alpha=1) 
                                     for x in range(len(sigmoid_params))]

    def offline_train(self, inputs, outputs):
        '''
        Function to execute the offline training of the RHONN, with a 
        pre-defined training set.
        
        Parameters
        ----------
        inputs - an iterable with the inputs given to the system in a simulation 
                 with k as a simulation time unit, with inputs[k] as an iterable.
        outputs - an iterable with the outputs measured from the system, where
                  each element is an iterable with the result from the system 
                  at the time k with inputs[k].
        '''
        # Initial values
        self.s_x = np.zeros((len(inputs), self.in_size))
        self.weights = np.random.rand(len(inputs), self.out_size, self.num_z)
        self.error = np.zeros((len(inputs), self.out_size))
        self.Z = np.random.rand(len(inputs), self.out_size, self.num_z)
        self.X_estimated = np.zeros((len(inputs), self.out_size))

        # Training
        for k in range(len(inputs)):
            # Applying activation functions
            for j in range(len(self.activation_funcs)):
                self.s_x[k, j] = self.activation_funcs[j](inputs[k, j])
            
            # Calculating network output
            if self.z_structure is not None:
                self.Z[k] = self.calculate_z(self.s_x[k])
            else:
                self.Z[k] = np.array([self.s_x[k] for x in range(self.num_z)])

            self.X_estimated[k] = [np.dot(self.weights[k][n].T, self.Z[k][n]) 
                                   for n in range(self.out_size)]

            # Calculating the error
            self.error[k] = np.subtract(outputs[k], self.X_estimated[k])
            
            if self.debug:
                print self.str_result.format(k, 
                                             outputs[k], 
                                             self.X_estimated[k], 
                                             self.error[k]) #DEBUG

            # Calculating the adjustment for all the array with EFK
            if self.fnc_weights_adjsmnt is None:
                weights_res = self.EKF_weights_adjustment(self.Z[k], 
                                                         self.weights[k], 
                                                         self.error[k])
                estimated_weights = weights_res[0]
                self.EKF_P = weights_res[1]
            else:
                estimated_weights = self.fnc_weights_adjsmnt(self.weights[k], 
                                                             self.error[k])
            
            
            # Save the new weights    
            if (k < len(inputs) - 1):
                self.weights[k+1] = estimated_weights
            else:
                self.weights[k] = estimated_weights

        # Ploting data
        if self.plot_data:
            fig_error = plt.figure('Error')
            for n in range(self.out_size):
                line = plt.plot(self.error[:, n], 
                                label='State var {}'.format(n+1))
            
            plt.legend()

            for n in range(self.out_size):
                fig = plt.figure('State variable {}'.format(n+1))
                line_1, = plt.plot(outputs[:, [n]], 
                                   label='System')
                line_2, = plt.plot(self.X_estimated[:, [n]], 
                                   linestyle='--', 
                                   label='RHONN', 
                                   color='red')
                plt.legend()

            plt.show()
        
    def inline_train(self, input, output):
        '''
        Function to execute the inline training of the RHONN
        
        Parameters
        ----------
        input - an iterable with the input given to the system.
        output - an iterable with the output measured from the system.
        '''
        # Initial values
        if not self.inline_started:
            self.s_x = np.zeros(self.in_size)
            self.weights = np.random.rand(self.out_size, self.num_z)
            self.error = np.zeros(self.out_size)
            self.Z = np.random.rand(self.out_size, self.num_z)
            self.X_estimated = np.zeros(self.out_size)
            self.k = 0
            
            # Var to save the training history
            self.inputs = []
            self.outputs = []
            self.net_outputs = []
            self.errors = []
            
            # Create figures for ploting data
            fig_error = plt.figure('Error')
            error_ax = fig_error.add_subplot(111)
            
            figs = [plt.figure('State variable {}'.format(n+1)) for n in range(self.out_size)]
            axes = [figs[n].add_subplot(111) for x in range(self.out_size)]
            
            self.inline_started = True

        self.inputs.append(input)
        self.outputs.append(output)
        
        # Applying activation functions
        for n in range(len(self.activation_funcs)):
            self.s_x[n] = self.activation_funcs[n](input[n])
        
        # Calculating network output
        if self.z_structure is not None:
            self.Z = self.calculate_z(self.s_x)
        else:
            self.Z = np.array([self.s_x for x in range(self.num_z)])

        self.X_estimated = [np.dot(self.weights[n].T, self.Z[n]) 
                               for n in range(self.out_size)]

        # Calculating the error
        self.error = np.subtract(output, self.X_estimated)
        
        # Save results
        self.net_outputs.append(self.X_estimated)
        self.errors.append(self.error)
        
        if self.debug:
            print self.str_result.format(k, 
                                         output, 
                                         self.X_estimated, 
                                         self.error)

        # Calculating the adjustment for all the array with EFK
        if self.fnc_weights_adjsmnt is None:
            weights_res = self.EKF_weights_adjustment(self.Z, 
                                                     self.weights, 
                                                     self.error)
            estimated_weights = weights_res[0]
            self.EKF_P = weights_res[1]
        else:
            estimated_weights = self.fnc_weights_adjsmnt(self.weights, 
                                                         self.error)
        
        
        # Save the new weights    
        self.weights = estimated_weights
        
        # Save the time units elapsed
        self.k += 1

        # Ploting data
        if self.plot_data:            
            plt.clear()
            
            for n in range(self.out_size):
                line = error_ax.plot(self.error[:, n], 
                                label='State var {}'.format(n+1))
            
            plt.legend()

            for n in range(self.out_size):
                axes[n].clear()
                line_1, = axes[n].plot(self.outputs[:, [n]], 
                                   label='System')
                line_2, = axes[n].plot(self.net_out[:, [n]], 
                                   linestyle='--', 
                                   label='RHONN', 
                                   color='red')
                plt.legend()

            plt.show()
        pass

if __name__ == '__main__':
    inputs_file = open('Training Set V2\\inputs.csv').read()
    outputsfile = open('Training Set V2\\outputs.csv').read()
    
    # Load inputs from file
    inputs = inputs_file.split('\n')
    if inputs[-1] == '':
        del(inputs[-1])

    for i in range(len(inputs)):
        temp = inputs[i].split(',')
        inputs[i] = map(lambda x: np.array(float(x)), temp)

    # Load outputs from file
    outputs = outputsfile.split('\n')
    if outputs[-1] == '':
        del(outputs[-1])

    for i in range(len(outputs)):
        temp = outputs[i].split(',')
        outputs[i] = map(lambda x: np.array(float(x)), temp)

    # Use inputs and outputs as a numpy array
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    # Parameters for the RHONN
    in_size = len(inputs[0])
    out_size = len(outputs[0])
    num_z = 3

    Z_structure = [ [[(1, 1)], [(2, 1)], [(2, 1)]],  # Z1
                    [[(1, 2)], [(2, 2)], [(2, 1)]],  # Z2
                    [[(1, 3)], [(2, 1)], [(2, 2)]] ] # Z3
    
    R = np.array([0.1, 0.01, 0.1])
    P = [np.identity(num_z) for x in range(out_size)]
    Q = [np.identity(num_z) for x in range(out_size)]
    etas = np.array([0.8, 0.8, 1.5])

    EKF_params = (R, P, Q, etas)
    sigmoids_params = [(0.5, 1), (0.1, 1)]

    # Test the RHONN identifier
    test_rhonn = RHONN((in_size, out_size, num_z), z_structure=Z_structure)
    test_rhonn.training_params(sigmoid_params=sigmoids_params, 
                               EKF_parameters=EKF_params)
    test_rhonn.offline_train(inputs, outputs)