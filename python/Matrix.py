
import pandas as pd
import random
import numpy as np 
import math
#from numba import vectorize
#from numba import cuda, float32
#from pycuda import driver, compiler, gpuarray, tools
#import pycuda.autoinit

kernel_code_template = """
__global__ void MatrixAddKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < rows  && ty < columns )
    c[ty * columns + tx] = a[ty * columns  + tx]  + b[ty * columns  + tx] ;

}
__global__ void MatrixSubtractKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < rows  && ty < columns )
    c[ty * columns + tx] = a[ty * columns  + tx]  - b[ty * columns + tx] ;

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
__global__ void MatrixMulKernel(int rows, int columns, float *a, float *b, float *c)
{
    int tx = threadIdx.x  + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    float Pvalue = 0;

        if (tx < rows  && ty < columns ) {
    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
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
    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.  
       for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
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

        if False and Matrix.matrix_initialised == False :
            Matrix.matrix_initialised  = True
            kernel_code = kernel_code_template % {
                }
            mod = compiler.SourceModule(kernel_code)
            Matrix.gpu_matrixaddkernel = mod.get_function("MatrixAddKernel")
            Matrix.gpu_matrixsubtractkernel = mod.get_function("MatrixSubtractKernel")
            Matrix.gpu_multiplyscalarkernel = mod.get_function("MultiplyScalarKernel")
            Matrix.gpu_multiplymatrixkernel = mod.get_function("MatrixMulKernel")
            Matrix.gpu_multiplyTransmatrixkernel = mod.get_function("TransposeMatrixMulKernel")
            Matrix.gpu_mapactivationkernel = mod.get_function("MapActivationKernel")
            Matrix.gpu_mapDeactivationkernel = mod.get_function("MapDeActivationKernel")
            Matrix.gpu_mapCopykernel = mod.get_function("CopyKernel")
            Matrix.gpu_mapCopyTransposekernel = mod.get_function("CopyTransposeKernel")
            
            print("Initialised Kernels for GPU ")

        self.gpu_enabled = gpu_enabled
        
        if vector is not None:
            self.rows = np.int32(1)
            self.cols = np.int32(len(vector))
            self.mat = np.array([[random.random() for col in range(self.cols)] for row in range(self.rows)])
            for i in range(self.cols):
                self.mat[0][i] = vector[i]
            return
        self.rows = np.int32(rows)
        self.cols = np.int32(cols)
        self.mat = np.array([[random.random() for col in range(self.cols)] for row in range(self.rows)],dtype='f')
        
        # TODO only for testing remove later
        if False:
            k=1
            for i in range(self.rows):
                for j in range(self.cols):
                    self.mat[i][j] = k
                    k=k+1
                
        if self.gpu_enabled :
            self.mat_gpu = gpuarray.to_gpu(self.mat) 
            self.gpu_blockx = 20
            self.gpu_blocky = 20
            self.gpu_block= (np.int(20),np.int(20),np.int(1))
            self.gpu_number_of_blocks_x = np.int(self.rows/self.gpu_blockx)+1
            self.gpu_number_of_blocks_y = np.int(self.rows/self.gpu_blocky)+1
            self.gpu_grid = (np.int(self.rows/self.gpu_blockx)+1,np.int(self.rows/self.gpu_blockx)+1,np.int(1))
        
    def getFirstElement(self):
        return self.mat[0][0] 
                
    def injest(self, X ):
        if self.cols > 1:
            print("ERROR in Injest: More then columns") 
        for i in range(self.rows):
            self.mat[i][0] = X[i]
                
    def colView(self, c):
        newmat =  Matrix(1,self.rows,None)
        for i in range(self.rows):
            newmat.mat[0][i] = self.mat[i][c]
        return newmat
    
    def add(self, X,Y):
        if self.gpu_enabled :
            self.gpu_matrixaddkernel(
                self.rows,self.cols,
                X.mat_gpu, Y.mat_gpu, 
                self.mat_gpu, 
                block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x, self.gpu_number_of_blocks_y,1),
                #block = self.gpu_block, grid = self.gpu_grid,
                )
        else:            
            self.mat = X.mat + Y.mat
        
    def mapActivation(self):
        if self.activation == "sigmoid" :
            if self.gpu_enabled :
                self.gpu_mapactivationkernel(
                    self.mat_gpu,
                    block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x+1, self.gpu_number_of_blocks_x+1,1),
                    )
            else:
                self.mat = 1 / (1 + np.exp(-self.mat))
        else:
            self.mat = np.tanh(self.mat)
 
    def compositeActivation(self, A,B):
        # add + mapActivation
        self.add(A,B)
        self.mapActivation()
                    
    def compositeDeActivation(self, A,B,s):
        # copy + deactivation+MultiplyScalar
        self.mat = A.mat
        self.mapDeActivation()
        self.multiplyScalar(B, s)
        
    def compositeMultiplyTranspose(self, A,B,C):
        # MultiplyTranspose + add
        C.multiplyTranspose(A,B)
        self.add(self, C)
     
                
    def mapDeActivation(self):
        if self.activation == "sigmoid" :
            if self.gpu_enabled :
                self.gpu_mapDeactivationkernel(
                    self.mat_gpu,
                    block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x+1, self.gpu_number_of_blocks_x+1,1),
                    )
            else:
                self.mat =  self.mat * (1 - self.mat)
        else:
            self.mat = (1.0 - (self.mat*self.mat))
                    
    def subtract(self, X,Y):
        if self.gpu_enabled :
            self.gpu_matrixsubtractkernel(
                X.mat_gpu, Y.mat_gpu, 
                self.mat_gpu, 
                block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x+1, self.gpu_number_of_blocks_x+1,1),
                )      
        else:           
            self.mat = X.mat - Y.mat
            
    def copy(self, X):
        if self.gpu_enabled :
            #self.mat_gpu.set(X.mat)
            self.gpu_mapCopykernel(
                X.mat_gpu, 
                self.mat_gpu, 
                block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x+1, self.gpu_number_of_blocks_x+1,1),
                ) 
        else:
            self.mat = X.mat

    def copyTranspose(self, X):
        if self.gpu_enabled :
            self.gpu_mapCopyTransposekernel(
                X.mat_gpu, 
                self.mat_gpu, 
                block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x+1, self.gpu_number_of_blocks_x+1,1),
                )
        else:
            self.mat = X.mat.T        
                
    def multiplyScalar(self,X, s):
        if self.gpu_enabled :
            self.gpu_multiplyscalarkernel(
                # TODO:  convert s to suitable
                X.mat_gpu, s, 
                self.mat_gpu,
                block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x+1, self.gpu_number_of_blocks_x+1,1),
                )
        else:
            self.mat = self.mat*X.mat*s

    def multiply(self, X,Y):
        if self.gpu_enabled :
            self.gpu_multiplymatrixkernel(
                X.mat_gpu, Y.mat_gpu,
                self.mat_gpu,
                block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x+1, self.gpu_number_of_blocks_x+1,1),
                ) 
        else:
            self.mat = np.matmul(X.mat,Y.mat)

    def multiplyTranspose(self, X,Y):
        if X.cols != Y.cols:
            print("ERROR in MultiplicationTranspose")
        
        if self.gpu_enabled :
            self.gpu_multiplyTransmatrixkernel(
                X.mat_gpu, Y.mat_gpu,
                self.mat_gpu,
                block = (self.gpu_blockx, self.gpu_blocky, 1), grid = (self.gpu_number_of_blocks_x+1, self.gpu_number_of_blocks_x+1,1),
                ) 
        else:
            self.mat = np.matmul(X.mat,Y.mat.T)
 
    def multiplyTranspose2(self, X,Y):
        if X.rows != Y.rows:
            print("ERROR in MultiplicationTranspose2")

        self.mat = np.matmul(X.mat.T,Y.mat)      