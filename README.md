 
# Performance of NeuralNet on various Platform:

Following are different performance experiments of Neuralnet used for prediction and Training purposes:
- NeuralNet using golang
- NeuralNet using different versions of Python
    - using List : very slow (260sec)
    - using Arrays: Better then List (160sec)
    - using numpy: Best of all the three (95sec)
- NeuralNet using CPU versus GPU
    - Small number of neurons vs large number of neurons per layer:
      -  CPU latency is better if the  number of neurons are less , but GPU as better latency if the neurons are large in number. The reason is overhead in submitting the job to GPU.
    - Workload: Training vs predict:  
      -  GPU provides better latency in Training vs Predict.
    -  Batch vs Without Batch: Batch is good for GPU. 
    

# Tests 

<table border=1>
<thead>
<tr>
<th>Test-no</th>
<th>Description</th>
<th>Result</th>
</tr>
</thead>
<tbody>
<tr>
<th>1</th>
<th> with GPU
</th>
<th>time taken for trainning=5.4sec 
 cpu utilization=100% gpu=100%
</th>
</tr>
<tr>
<th>2</th>
<th> with CPU
   </th>
<th>time taken for trainning=16sec  cpu utilization=500% gpu utilization=0%</th>
</tr>
</tbody></table>

# overhead of Syscalls 

 ```
 % time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 55.93    1.908009           2   1240313           sched_yield
 39.68    1.353411       22939        59           poll
  4.14    0.141347        1178       120           clock_gettime
  0.25    0.008448        8448         1           restart_syscall
  0.00    0.000000           0        50         6 futex
------ ----------- ----------- --------- --------- ----------------
```

