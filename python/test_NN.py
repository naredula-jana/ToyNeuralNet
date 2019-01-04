from NeuralNetV4 import *
import cProfile
import sys
import time

pr = cProfile.Profile()
pr.enable()

nn = NeuralNet([4,1600,2600,1600,2500,1],True)
#nn = NeuralNet([4,3,4,5,1],False)
data = pd.read_csv("./iris_training.csv")
#print (data)
sepallength = data.SepalLength.values.tolist()
sepalwidth = data.SepalWidth.values.tolist()
petallength = data.PetalLength.values.tolist()
petalwidth = data.PetalWidth.values.tolist()
types = data.types.values.tolist()

datalen = len(sepallength)
i=0
while i< datalen :
    if types[i] == 2:
        types[i] = 0.5
    nn.bulk_injest([sepallength[i],sepalwidth[i],petallength[i],petalwidth[i]],types[i])
    i=i+1
    
#nn.bulk_print()
#sys.exit()

iteration=0
nn.debug=False
start = time.time()
while iteration< 50000 :
    i=0
    if iteration%20 == 0:
        #nn.debug=False
        #nn.data_index=iteration
        print(iteration," Error Percentage: ",nn.error_percentage)
        end = time.time()
        print("Time taken: ",(end-start))
        start = time.time()
    while i< datalen :
        if types[i] == 2:
            types[i] = 0.5
        nn.train([sepallength[i],sepalwidth[i],petallength[i],petalwidth[i]],[types[i]],1,i)
        i=i+1
        if i>2 and nn.debug==True:
            sys.exit()
    if iteration%2000 == 0:
        nn.debug=False
    iteration=iteration+1
    
pr.disable()
#pr.print_stats()
