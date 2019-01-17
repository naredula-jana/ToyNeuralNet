 
# Performance of NeuralNet on various Platform:

Following are different performance experiments of Neuralnet used for prediction and Training purposes:
- NeuralNet using golang
- NeuralNet using different versions of Python
    - using python List : very slow (260sec)
    - using python Arrays: Better then List (160sec)
    - using numpy library: Best of all the three (95sec) : Reason: Array are implemented natively, due to this matrix multiplication is faster.
    - using GPU : better then numpy when the layer contain large number of neurons.
- NeuralNet using CPU versus GPU
    - Small number of neurons vs large number of neurons per layer:
      -  CPU latency is better if the  number of neurons are less , but GPU as better latency if the neurons are large in number. The reason is overhead in submitting the job to GPU.
    - Workload: Training vs predict:  
      -  GPU provides better latency in Training  when compare to Predict. Reason: Predict contain mXn by mX1 multiplication, but Trainning need mXn by mXn mulitplication , and also update of weights.
    -  Batch vs Without Batch: 
    	 - Batch is good for GPU and CPU. As batch level increases, the efficiency of cpu parallelism goes up. 
    

# Tests 

 NeuralNet layers size: [4,1600,2600,1600,2500,1]
 learning_rate=0.005
 
 
<table border=1>
<thead>
<tr>
<th>Test-no</th>
<th>Description</th>
<th>Result</th>
</tr>
<tr>
<th>1</th>
<th> with CPU batch=3
   </th>
<th>time taken for trainning=24sec  cpu utilization=600% gpu utilization=0%</th>
</tr>
<tr>
<th>2</th>
<th> with GPU batch=3
</th>
<th>time taken for trainning=9sec 
 cpu utilization=100% gpu=100%
</th>
</tr>

<tr>
<th>3</th>
<th> with CPU batch=10
   </th>
<th>time taken for trainning=9sec  cpu utilization=600% gpu utilization=0%</th>
</tr>
<tr>
<th>4</th>
<th> with GPU batch=10
</th>
<th>time taken for trainning=5.4sec 
 cpu utilization=100% gpu=100%
</th>
</tr>
</tbody></table>

# overhead of Syscalls 

 ```
 CPU TEST: 600% CPU : most of time spend in matrix multiplication
 system calls per second in CPU test: 620k /sec
 % time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 55.93    1.908009           2   1240313           sched_yield
 39.68    1.353411       22939        59           poll
  4.14    0.141347        1178       120           clock_gettime
  0.25    0.008448        8448         1           restart_syscall
  0.00    0.000000           0        50         6 futex
------ ----------- ----------- --------- --------- ----------------

GPU TEST: 100% CPU  :  Most of the time spend in IO with GPU due to this time for futex  are high.
 system calls per second in CPU test: 100k /sec, Here mostly IO instensive with GPU.
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 41.52    5.524696       21923       252           poll
 40.11    5.337111       53910        99        49 futex
 17.40    2.314710           1   2109500           clock_gettime
  0.68    0.090668       45334         2         1 restart_syscall
  0.30    0.039534           1     36481           getpid
  0.00    0.000029           3        10           write
------ ----------- ----------- --------- --------- ----------------
100.00   13.306748               2146344        50 total
```

GPU spec:

```
For GPU test
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.48                 Driver Version: 390.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GT 1030     Off  | 00000000:01:00.0  On |                  N/A |
| 55%   54C    P0    N/A /  30W |    495MiB /  2000MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1242      G   /usr/lib/xorg/Xorg                            18MiB |
|    0      1284      G   /usr/bin/gnome-shell                          48MiB |
|    0      3392      G   /usr/lib/xorg/Xorg                           125MiB |
|    0      3548      G   /usr/bin/gnome-shell                          80MiB |
|    0      6291      G   ...-token=E69B0CC02C6F5687670BDDD2298FD240    80MiB |
|    0      7004      C   python                                       131MiB |
+-----------------------------------------------------------------------------+
```
top:  for CPU TEST : cpu consumption: 600%

```
top - 11:28:28 up  2:28,  3 users,  load average: 1.49, 1.98, 2.02
Tasks: 302 total,   2 running, 240 sleeping,   0 stopped,   0 zombie
%Cpu(s): 99.1 us,  0.9 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.1 si,  0.0 st
KiB Mem : 16354764 total, 12074680 free,  1796652 used,  2483432 buff/cache
KiB Swap:        0 total,        0 free,        0 used. 14183740 avail Mem 

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND                                                                                                                                             
 6930 a         20   0 13.431g 339968 121916 R 595.7  2.1   1:18.84 python
 ```