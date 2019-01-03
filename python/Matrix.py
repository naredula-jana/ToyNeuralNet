
import pandas as pd
import random
import numpy as np 
import math
#from numba import vectorize
#from numba import cuda, float32
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

kernel_code_template = """
#include <stdio.h>
#include<stdlib.h>
__global__ void MatrixAddKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < rows  && ty < columns )
    c[tx * columns  + ty] = a[tx * columns  + ty]  + b[tx * columns  + ty] ;

}
__global__ void MatrixSubtractKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < rows  && ty < columns )
    c[tx * columns  + ty] = a[tx * columns  + ty]  - b[tx * columns  + ty] ;

}
__global__ void MultiplyScalarKernel(int rows, int columns, float *a, float scalar, float *b)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (tx < rows  && ty < columns )
    b[ty * columns + tx] = (b[ty * columns + tx]) * (a[ty * columns  + tx]) * (scalar);
}
__global__ void MapActivationKernel(int rows, int columns, float *a)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
     if (tx < rows  && ty < columns )
    a[ty * columns + tx] = 1/(1+ exp(-a[ty * columns + tx])) ;
}
__global__ void CopyKernel(int rows, int columns, float *a, float *b)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (tx < rows  && ty < columns )
    b[ty * columns + tx] = a[ty * columns + tx] ;
}
__global__ void CopyTransposeKernel(int rows, int columns, float *a, float *b)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (tx < rows  && ty < columns )
    b[ty * columns + tx] = a[tx * columns + ty] ;
}

__global__ void MapDeActivationKernel(int rows, int columns, float *a)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
     if (tx < rows  && ty < columns )
    a[ty * columns + tx] = a[ty * columns+ tx] * (1-a[ty * columns+ tx]) ;
}


__global__ void CompositeDeActivationKernel(int rows, int columns, float *a, float *b, float scalar, float *c )
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    // copy + deactivation+MultiplyScalar   self=A, deactivation , self= self*B*s
    float Output = 0;
    
    if (tx < rows  && ty < columns ) {
    // 1.copy
        Output = a[tx * columns  + ty];
        
    //2. Deactivation
        Output = Output * (1-Output) ;
        
    //3. Multiply with b and scalar
        c[tx * columns  + ty] = (Output) * (b[tx * columns  + ty]) * (scalar);
    }
}
__global__ void CompositeActivationKernel(int rows, int common_cols, int columns, float *a, float *b, float *c, float*d)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
    //  # multiply + add + mapActivation : output = (A*B) + C  , activation
    // each thread picks i and j of the matrix
    //        self.mat = np.matmul(A.mat,B.mat)
    //        self.add(self,C)
    //        self.mapActivation()

    float Output = 0;

    if (tx < rows  && ty < columns ) {    
    // 1.Multiplication
       for (int k = 0; k < common_cols; ++k) {
          float Aelement = a[tx * common_cols + k];
          float Belement = b[k * columns + ty];
          Output += Aelement * Belement;
       } 
 
    // 2. Addition
       Output = Output  + c[tx * columns  + ty] ;
       
    //3. Aactivation 
      d[tx * columns + ty] = 1/(1+ exp(-Output)) ; 
    }
}
__global__ void CompositeTransposeMulKernel(int rows, int common_cols, int columns, float *a, float *b, float *c, float *d)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    
//MultiplyTranspose + add ,  output= A*B.T + C
    float Output = 0;

    if (tx < rows  && ty < columns ){
       for (int k = 0; k < common_cols; ++k) {
          float Aelement = a[tx * common_cols + k];
          float Belement = b[ty * common_cols + k];
          Output += Aelement * Belement;
       }
       //d[tx * columns + ty] = Output + c[tx * columns + ty];
       // NOT Needed c[tx * columns + ty] = Output
       d[tx * columns + ty] =  d[tx * columns + ty] + Output;
    }
}
__global__ void Transpose2MatrixMulKernel(int rows, int common_cols,  int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    float Pvalue = 0;

    if (tx < rows  && ty < columns ) {
       for (int k = 0; k < common_cols; ++k) {
          float Aelement = a[k * rows + tx];
          float Belement = b[k * columns + ty];
          Pvalue += Aelement * Belement;
       }
       c[tx * columns + ty] = Pvalue;
    }
}
__global__ void MatrixMulKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    float Pvalue = 0;

    if (tx < rows  && ty < columns ) {
       for (int k = 0; k < columns; ++k) {
          float Aelement = a[ty * columns + k];
          float Belement = b[k * rows + tx];
          Pvalue += Aelement * Belement;
       }
       c[ty * columns + tx] = Pvalue;
    }
}
__global__ void TransposeMatrixMulKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    float Pvalue = 0;

    if (tx < rows  && ty < columns ){
       for (int k = 0; k < columns; ++k) {
          float Aelement = a[ty * columns + k];
          float Belement = b[tx * columns + k];
          Pvalue += Aelement * Belement;
       }
       c[ty * columns + tx] = Pvalue;
    }
}

"""

class Matrix:
    activation = "sigmoid"
    gpu_matrixadd = 1
    matrix_initialised = False      
  
    def __init__(self, rows,cols,gpu_enabled,vector):
        
        if True and Matrix.matrix_initialised == False :
            Matrix.matrix_initialised  = True
            kernel_code = kernel_code_template % {
                }
            mod = compiler.SourceModule(kernel_code)
            
            #Matrix.gpu_multiplyscalarkernel = mod.get_function("MultiplyScalarKernel")
            #Matrix.gpu_multiplymatrixkernel = mod.get_function("MatrixMulKernel")
            #Matrix.gpu_multiplyTransmatrixkernel = mod.get_function("TransposeMatrixMulKernel")
            #Matrix.gpu_mapactivationkernel = mod.get_function("MapActivationKernel")
            #Matrix.gpu_mapDeactivationkernel = mod.get_function("MapDeActivationKernel")
            #Matrix.gpu_mapCopykernel = mod.get_function("CopyKernel")
            #Matrix.gpu_mapCopyTransposekernel = mod.get_function("CopyTransposeKernel")
            
            Matrix.gpu_matrixaddkernel = mod.get_function("MatrixAddKernel")
            Matrix.gpu_matrixSubtractkernel = mod.get_function("MatrixSubtractKernel")
            Matrix.gpu_multiplyTrans2matrixkernel = mod.get_function("Transpose2MatrixMulKernel")
            
            Matrix.gpu_compositeActivationkernel = mod.get_function("CompositeActivationKernel")
            Matrix.gpu_compositeDeactivationkernel = mod.get_function("CompositeDeActivationKernel")
            Matrix.gpu_compositeTransposeMulkernel = mod.get_function("CompositeTransposeMulKernel")
            
            print("Initialised Kernels for GPU ")

        self.gpu_enabled = gpu_enabled
        
        if vector is not None:
            self.rows = np.int32(1)
            self.cols = np.int32(len(vector))
            self.mat = np.array([[random.random() for col in range(self.cols)] for row in range(self.rows)],dtype='f')
            if True:
                k=1
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.mat[i][j] = k
                        self.mat[i][j] = 1
                        k=k+1
                        
            for i in range(self.cols):
                self.mat[0][i] = vector[i]
        else:
            self.rows = np.int32(rows)
            self.cols = np.int32(cols)
            self.mat = np.array([[random.random() for col in range(self.cols)] for row in range(self.rows)],dtype='f')
            #TODO :change below type to float32 for gpu
            #self.mat = np.random.randn(rows,cols)*np.sqrt(2/cols)
            
            # TODO only for testing remove later
            if True:
                k=1
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.mat[i][j] = k*0.25
                        #self.mat[i][j] = 3
                        k=k+1
                
        if self.gpu_enabled :
            self.mat_gpu = gpuarray.to_gpu(self.mat) 
            self.gpu_blockx = 25
            self.gpu_blocky = 25
            self.gpu_block= (np.int(self.gpu_blockx ),np.int(self.gpu_blocky),np.int(1))
            #self.gpu_number_of_blocks_x = np.int(self.rows/self.gpu_blockx)+1
            #self.gpu_number_of_blocks_y = np.int(self.cols/self.gpu_blocky)+1
            self.gpu_grid = (np.int(self.rows/self.gpu_blockx)+1,np.int(self.cols/self.gpu_blockx)+1,np.int(1))
        
    def getFirstElement(self):
        if self.gpu_enabled :
            ret=self.mat_gpu.get()
            return ret[0][0]
        else:
            return self.mat[0][0] 
                
    def injest(self, X ):
        if self.cols > 1:
            print("ERROR in Injest: More then columns") 
        for i in range(self.rows):
            self.mat[i][0] = X[i]
            
        if self.gpu_enabled :
            self.mat_gpu.set(self.mat)
                

    
    def add(self, X,Y):
        if self.gpu_enabled :
            self.gpu_matrixaddkernel(
                self.rows,self.cols,
                X.mat_gpu, Y.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:            
            self.mat = X.mat + Y.mat
        
    def mapActivation(self):
        if self.activation == "sigmoid" :
            if self.gpu_enabled :
                self.gpu_mapactivationkernel(
                    self.mat_gpu,
                    block = self.gpu_block, grid = self.gpu_grid,
                    )
            else:
                self.mat = 1 / (1 + np.exp(-self.mat))
        else:
            self.mat = np.tanh(self.mat)
 
    def compositeActivation(self,A,B,C):
        # multiply + add + mapActivation : output = (A*B) + C
        if self.gpu_enabled :
            self.gpu_compositeActivationkernel(
                self.rows, A.cols, self.cols,
                A.mat_gpu, B.mat_gpu,C.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = np.matmul(A.mat,B.mat)
            self.add(self,C)
            self.mapActivation()
                    
    def compositeDeActivation(self, A,B,s):
        # copy + deactivation+MultiplyScalar   self=A, deactivation , self= self*B*s
        if self.gpu_enabled :
            v = np.float32(s)
            self.gpu_compositeDeactivationkernel(
                self.rows,self.cols,
                A.mat_gpu, B.mat_gpu, v , 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = A.mat
            self.mapDeActivation()
            self.mat = self.mat*B.mat*s
        
    def compositeMultiplyTranspose(self, A,B,C):
        if A.cols != B.cols:
            print("ERROR in MultiplicationTranspose")
        # MultiplyTranspose + add ,  output= A*B.T + C
        if self.gpu_enabled :
            self.gpu_compositeTransposeMulkernel(
                self.rows, A.cols, self.cols,
                A.mat_gpu, B.mat_gpu, C.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:        
            #C.multiplyTranspose(A,B)
            C.mat = np.matmul(A.mat,B.mat.T)
            self.add(self, C)
     
                
    def mapDeActivation(self):
        if self.activation == "sigmoid" :
            if self.gpu_enabled :
                self.gpu_mapDeactivationkernel(
                    self.mat_gpu,
                    block = self.gpu_block, grid = self.gpu_grid,
                    )
            else:
                self.mat =  self.mat * (1 - self.mat)
        else:
            self.mat = (1.0 - (self.mat*self.mat))
                    
    def subtract(self, X,Y):
        if self.gpu_enabled :
            #print("gpu subtract: ",X.mat_gpu.get())
            #print("gpu subtract: ",Y.mat_gpu.get())
            self.gpu_matrixSubtractkernel(
                self.rows,self.cols,
                X.mat_gpu, Y.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )     
        else:           
            self.mat = X.mat - Y.mat

    def multiplyTranspose2(self, X,Y):
        if X.rows != Y.rows:
            print("ERROR in MultiplicationTranspose2")
            
        if self.gpu_enabled :
            #print(" common rows: ",X.rows)
            self.gpu_multiplyTrans2matrixkernel(
                self.rows,X.rows,self.cols,
                X.mat_gpu, Y.mat_gpu,
                self.mat_gpu,
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = np.matmul(X.mat.T,Y.mat) 
            
    def printMat(self,s):
        if self.gpu_enabled :
            print("gpu Matrix: ",s,self.mat_gpu.get())
        else:
            print("Matrix: ", s,self.mat)
                  
# -------------------- NOT in use functions
'''
             
    def NOTINUSE_copy(self, X):
        if self.gpu_enabled :
            #self.mat_gpu.set(X.mat)
            self.gpu_mapCopykernel(
                self.rows,self.cols,
                X.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                ) 
        else:
            self.mat = X.mat

    def NOTINUSE_copyTranspose(self, X):
        if self.gpu_enabled :
            self.gpu_mapCopyTransposekernel(
                self.rows,self.cols,
                X.mat_gpu, 
                self.mat_gpu, 
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = X.mat.T        
                
    def NOTINUSE_multiplyScalar(self,X, s):
        if self.gpu_enabled :
            v =  np.float32(s)
            self.gpu_multiplyscalarkernel(
                self.rows,self.cols,
                X.mat_gpu, v, 
                self.mat_gpu,
                block = self.gpu_block, grid = self.gpu_grid,
                )
        else:
            self.mat = self.mat*X.mat*s

    def NOTINUSE_multiply(self, X,Y):
        if self.gpu_enabled :
            print("ERROR :  NOTINUSE")
            self.gpu_multiplymatrixkernel(
                self.rows,self.cols,
                X.mat_gpu, Y.mat_gpu,
                self.mat_gpu,
                block = self.gpu_block, grid = self.gpu_grid,
                ) 
        else:
            self.mat = np.matmul(X.mat,Y.mat)

    def NOTINUSE_multiplyTranspose(self, X,Y):
        if X.cols != Y.cols:
            print("ERROR in MultiplicationTranspose")
        
        if self.gpu_enabled :
            self.gpu_multiplyTransmatrixkernel(
                self.rows,self.cols,
                X.mat_gpu, Y.mat_gpu,
                self.mat_gpu,
                block = self.gpu_block, grid = self.gpu_grid,
                ) 
        else:
            self.mat = np.matmul(X.mat,Y.mat.T)
 
    def NOTINUSE_colView(self, c):
        newmat =  Matrix(1,self.rows,None)
        for i in range(self.rows):
            newmat.mat[0][i] = self.mat[i][c]
        return newmat        
'''     