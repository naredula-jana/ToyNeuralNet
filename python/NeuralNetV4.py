from Matrix import *
import pandas as pd
import random
import numpy as np 
                 
class Layer:
    def __init__(self, curr_lay,prev_lay):    
        gpu_enabled = False   
         
        self.nodes = curr_lay
        self.weights = Matrix(curr_lay,prev_lay,gpu_enabled, None)
        
        #initialise the weights accordingly
        self.weights.mat = np.random.randn(curr_lay,prev_lay)*np.sqrt(2/prev_lay)
        #print("New weights :",curr_lay," :",self.weights.mat)
        
        self.weights_T = Matrix(prev_lay, curr_lay, gpu_enabled , None)
        self.weights_delta = Matrix(curr_lay,prev_lay, gpu_enabled, None)
        self.error_val =  Matrix(curr_lay,1,gpu_enabled , None) 
        self.output =  Matrix(curr_lay,1, gpu_enabled, None) 
        self.output_T =  Matrix(1, curr_lay, gpu_enabled, None)
        self.bias =  Matrix(curr_lay,1, gpu_enabled, None) 
        self.gradients = Matrix(curr_lay,1, gpu_enabled, None) 
         
class NeuralNet:
    def __init__(self, layers):
        self.layercount = len(layers)
        self.layers = []
        self.learning_rate = 0.04
        self.errorval = 0
        self.max_batch_size = 1
        self.debug = False
        self.data_index = 1
        
        self.cumulative_error =0 # sum of error so far
        self.data_count=0
        self.error_percentage=100
        
        prev_size = 1
        for curr_size in layers:
            self.layers.append(Layer(curr_size,prev_size))
            prev_size = curr_size
        print ("NeuralNetV3 created:  layers:",self.layercount," Activation FUNC : ",Matrix.activation)
        
        
    def predict(self, input_args, target_args, batch_size):
        batch_size = 1
        target_mat = Matrix(1,1,False, target_args)
        self.layers[0].output.injest(input_args)
            
        i = 1
        while i < (self.layercount):   
            #print(" type: ",type(self.layers[i].weights.mat),"  sectype: ",type(self.layers[i-1].output.mat))
            self.layers[i].output.multiply(self.layers[i].weights, self.layers[i-1].output)
            
            # TODO: Merge the below two : add+mapActivation
            if False:
                self.layers[i].output.add(self.layers[i].output,self.layers[i].bias)
                self.layers[i].output.mapActivation()
            else:
                self.layers[i].output.compositeActivation(self.layers[i].output,self.layers[i].bias)
                
            i = i+1
                
        last_layer = self.layercount-1
        self.layers[last_layer].error_val.subtract(target_mat, self.layers[last_layer].output)
        self.errorval = self.layers[last_layer].error_val.getFirstElement()
        
        # Stats : calculate the error
        self.cumulative_error = self.cumulative_error + abs(self.layers[last_layer].output.getFirstElement() -target_mat.getFirstElement())
        self.data_count = self.data_count +1
        self.error_percentage = (self.cumulative_error/self.data_count)*100
       
        
    def train(self, input_args, target_args, batch_size):
        self.predict(input_args,target_args,batch_size)

        if self.debug==True :
            print(self.data_index," Input:", input_args, "Target:", target_args, " : ",self.layers[self.layercount-1].output.mat[0][0])
            
        i = self.layercount-1
        while i>0 :
            # TODO:  Merge below 3: copy+deactivation+MultiplyScalar
            if False:
                self.layers[i].gradients.copy(self.layers[i].output)
                self.layers[i].gradients.mapDeActivation()
                self.layers[i].gradients.multiplyScalar(self.layers[i].error_val,self.learning_rate)
            else:
                self.layers[i].gradients.compositeDeActivation(self.layers[i].output,self.layers[i].error_val,self.learning_rate)

            # TODO: Merge Below Two: Multiply-Transpose + Add 
            if False:
                self.layers[i].weights_delta.multiplyTranspose(self.layers[i].gradients, self.layers[i-1].output)
                self.layers[i].weights.add(self.layers[i].weights, self.layers[i].weights_delta)
            else:
                self.layers[i].weights.compositeMultiplyTranspose(self.layers[i].gradients, self.layers[i-1].output, self.layers[i].weights_delta)
            
            self.layers[i].bias.add(self.layers[i].bias, self.layers[i].gradients)
            if (i-1)>0 :
                # TODO: merge below 2
                #self.layers[i].weights_T.copyTranspose(self.layers[i].weights)
                #self.layers[i-1].error_val.multiply(self.layers[i].weights_T, self.layers[i].error_val) 
                self.layers[i-1].error_val.multiplyTranspose2(self.layers[i].weights, self.layers[i].error_val)
            i = i-1
            


