from NeuralNetV1 import *
import cProfile

pr = cProfile.Profile()
pr.enable()

nn = NeuralNet([4,3,5,1])
data = pd.read_csv("../data/iris_training.csv")
#print (data)
sepallength = data.SepalLength.values.tolist()
sepalwidth = data.SepalWidth.values.tolist()
petallength = data.PetalLength.values.tolist()
petalwidth = data.PetalWidth.values.tolist()
types = data.types.values.tolist()

datalen = len(sepallength)

iteration=0
while iteration< 10000 :
    i=0
    if iteration%1000 == 0:
        #nn.debug=True
        #nn.data_index=iteration
        print(iteration," Error Percentage: ",nn.error_percentage)
    while i< datalen :
        if types[i] == 2:
            types[i] = 0.5
        nn.train([sepallength[i],sepalwidth[i],petallength[i],petalwidth[i]],[types[i]],1)
        i=i+1
    if iteration%10000 == 0:
        nn.debug=False
    iteration=iteration+1
    
pr.disable()
pr.print_stats()