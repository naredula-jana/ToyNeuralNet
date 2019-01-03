from Matrix import *
import pandas as pd
import random
import numpy as np 
                 
class Layer:
    def __init__(self, curr_lay,prev_lay,gpu_enabled):     
         
        self.nodes = curr_lay
        self.weights = Matrix(curr_lay,prev_lay,gpu_enabled, None)
        
        #initialise the weights accordingly
        #self.weights.mat = np.random.randn(curr_lay,prev_lay)*np.sqrt(2/prev_lay)
        #print("New weights :",curr_lay," :",self.weights.mat)
        
        
        self.weights_T = Matrix(prev_lay, curr_lay, gpu_enabled , None)
        self.weights_delta = Matrix(curr_lay,prev_lay, gpu_enabled, None)
        self.error_val =  Matrix(curr_lay,1,gpu_enabled , None) 
        self.output =  Matrix(curr_lay,1, gpu_enabled, None) 
        self.output_T =  Matrix(1, curr_lay, gpu_enabled, None)
        self.bias =  Matrix(curr_lay,1, gpu_enabled, None) 
        self.gradients = Matrix(curr_lay,1, gpu_enabled, None) 
         
class NeuralNet:
    def __init__(self, layers, gpu_enabled):
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
        self.gpu_enabled= gpu_enabled
        
        prev_size = 1
        for curr_size in layers:
            self.layers.append(Layer(curr_size,prev_size,gpu_enabled))
            prev_size = curr_size
        
        self.target_mat = Matrix(1,1,self.gpu_enabled, None)
        print ("NeuralNetV4 created:  layers:",self.layercount," Activation FUNC : ",Matrix.activation, "GPU enabled:",gpu_enabled)
        
        
    def predict(self, input_args, target_args, batch_size):
        batch_size = 1
        #target_mat = Matrix(1,1,self.gpu_enabled, target_args)
        self.target_mat.injest(target_args)
        self.layers[0].output.injest(input_args)
            
        i = 1
        while i < (self.layercount):  
            if self.debug==True :
                self.layers[i].weights.printMat(str(i)+" Pred-before weights") 
                self.layers[i-1].output.printMat(str(i-1)+" Pred-before output") 
                self.layers[i].bias.printMat(str(i)+" Pred-before bias")  
            self.layers[i].output.compositeActivation(self.layers[i].weights, self.layers[i-1].output, self.layers[i].bias)  
            if self.debug==True :
                self.layers[i].output.printMat(str(i)+" Pred-after output")   
            i = i+1
                
        last_layer = self.layercount-1
        if self.debug==True :
            self.layers[last_layer].error_val.printMat(str(0)+" sub-before error-val") 
            self.target_mat.printMat(str(0)+" sub-before target") 
            self.layers[last_layer].output.printMat(str(0)+" sub-before output") 
            
        self.layers[last_layer].error_val.subtract(self.target_mat, self.layers[last_layer].output)
        if self.debug==True :
            self.layers[last_layer].error_val.printMat(str(0)+" sub-after error_val") 
        self.errorval = self.layers[last_layer].error_val.getFirstElement()
        
        # Stats : calculate the error
        self.cumulative_error = self.cumulative_error + abs(self.layers[last_layer].output.getFirstElement() -self.target_mat.getFirstElement())
        self.data_count = self.data_count +1
        self.error_percentage = (self.cumulative_error/self.data_count)*100
       
        
    def train(self, input_args, target_args, batch_size):
        self.predict(input_args,target_args,batch_size)

        if self.debug==True :
            print(self.data_index," Input..:", input_args, "Target:", target_args, " : ",self.layers[self.layercount-1].output.getFirstElement())
            #self.layers[2].weights.printMat()
            
        i = self.layercount-1
        while i>0 :
            if self.debug==True :
                self.layers[i].error_val.printMat(str(i)+" before errorval") 
                self.layers[i].gradients.printMat(str(i)+" before gradient") 
                self.layers[i].output.printMat(str(i)+" before output") 
            
            self.layers[i].gradients.compositeDeActivation(self.layers[i].output,self.layers[i].error_val,self.learning_rate)

            if self.debug==True :
                self.layers[i].weights.printMat(str(i)+" before weights")
                self.layers[i].gradients.printMat(str(i)+" before gradients")
                self.layers[i-1].output.printMat(str(i)+" before output")
                self.layers[i].weights_delta.printMat(str(i)+" before weightsdelta")
            
            self.layers[i].weights.compositeMultiplyTranspose(self.layers[i].gradients, self.layers[i-1].output, self.layers[i].weights_delta)
            
            if self.debug==True :
                self.layers[i].weights.printMat(str(i)+" after weights")
                self.layers[i].gradients.printMat(str(i)+" after gradients")
                self.layers[i-1].output.printMat(str(i)+" after outputs")
                self.layers[i].weights_delta.printMat(str(i)+" after weightsdelta")
            
            self.layers[i].bias.add(self.layers[i].bias, self.layers[i].gradients)
            if (i-1)>0 :
                self.layers[i-1].error_val.multiplyTranspose2(self.layers[i].weights, self.layers[i].error_val)
            i = i-1
            


