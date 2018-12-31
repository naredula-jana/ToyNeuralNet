from Matrix import *
import time
import cProfile

pr = cProfile.Profile()
pr.enable()

# Here maximum 32X32=1024 of blocksize is allowed 
'''
root@HomeServer:/data/gpu_test# python ./test_matrix.py 
GPU Time taken:  48.32857346534729  /* cpu=100% */
CPU Time taken:  76.59190392494202   /* cpu=600% */
'''

r=1500
c=1600
mat1 = Matrix(r,c,True, None)
mat2 = Matrix(r,c,True, None)
mat3 = Matrix(r,c,True, None)

mat4 = Matrix(r,c,False, None)
mat5 = Matrix(r,c,False, None)

k=0
while k<10:
    mat4.mat[0][k] = 22222
    k=k+1
#mat3.copyTranspose(mat4)
#mat5.copyTranspose(mat4)
#print ("Matrix transpose copy:")
#print (mat3.mat_gpu.get())
#print (mat5.mat)

loops = 1000000
loops = 10000

start = time.time()
i =0
while i<loops:  
    #mat3.mapActivation()
    #mat3.mapDeActivation()
    #mat3.multiply(mat1,mat2)
    #mat3.multiplyTranspose(mat1, mat2)
    
    mat3.add(mat1,mat2)
    #mat3.multiplyScalar(mat3, 2)
    #mat3.multiplyScalar(mat3, 0.5)
    i=i+1
end = time.time()
print("GPU Time taken: ",(end-start))

start = time.time()
i =0
while i<loops:  
    #mat4.mapActivation()
    #mat4.mapDeActivation()
    
    #mat4.multiply(mat1, mat2)
    #mat4.multiplyTranspose(mat1, mat2)
    
    mat4.add(mat1,mat2)
    #mat4.multiplyScalar(mat4, 2)
    #mat4.multiplyScalar(mat4, 0.5)
    i=i+1
end = time.time()
print("CPU Time taken:. add. ",(end-start))

print ("-" * 80)
print ("Matrix C (GPU):")
print (mat3.mat_gpu.get())
print ("Matrix C CPU:")
print (mat4.mat)
pr.disable()
#pr.print_stats()