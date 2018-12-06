package main

import (
	"flag"
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"log"
	"os"
	"runtime/pprof"
	"strconv"
	"strings"
	"time"
)

var nn NeuralNet

func train_set(i int) {
	if i == 0 {
		train([]float64{0, 0}, []float64{0},1)
	}
	if i == 1 {
		train([]float64{0, 1}, []float64{1},1)
	}
	if i == 2 {
		train([]float64{1, 0}, []float64{1},1)
	}
	if i == 3 {
		train([]float64{1, 1}, []float64{0},1)
	}
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func strToinput(in string) ([]float64, float64) {
	slist := strings.Split(in, " ")
	if len(slist) == 5 {
		input := make([]float64, 4)

		input[0], _ = strconv.ParseFloat(slist[0], 64)
		input[1], _ = strconv.ParseFloat(slist[1], 64)
		input[2], _ = strconv.ParseFloat(slist[2], 64)
		input[3], _ = strconv.ParseFloat(slist[3], 64)
		target, _ := strconv.ParseFloat(slist[4], 64)
		return input, target / 2
	} else {
		fmt.Println(" error parsing input:", in)
	}
	return nil, 0
}

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if true {
		rawData, err := base.ParseCSVToInstances("data/iris_training.csv", false)
		if err != nil {
			panic(err)
		}

		//trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

		// Print a pleasant summary of your data.
		//fmt.Println("RAW DAAT :",rawData)
		fmt.Println(" 2nd ROW:", rawData.RowString(2))
		data, output := strToinput(rawData.RowString(2))
		fmt.Println(" data: ", data, " output:", output)
		nn = NeuralNet{}
		nn.Init([]int{4, 3, 1})
		for i := 0; i < 800000; i++ {
			nn.debug = false
			if i%10000 == 0 {
				nn.debug = true
			}
			data, output := strToinput(rawData.RowString(i % 73))
			train(data, []float64{output}, 1)
			if nn.debug {
				fmt.Println(i, " :---------")
			}
		}
	} else {
		nn = NeuralNet{}
		nn.Init([]int{2, 6, 4, 5, 1})
		startTime := time.Now()
		for i := 0; i < 80000; i++ {
			nn.debug = false
			if i%5000 == 0 {
				nn.debug = true
				//fmt.Println("errorval: ",nn.errorval)
			}
			train_set(i%4)
			train_set((i+1)%4)
			train_set((i+2)%4)
			train_set((i+3)%4)
			if nn.debug {
				fmt.Println(i, " :---------")
			}
		}
		endTime := time.Now()
		fmt.Println("Time took for Training :", endTime.Sub(startTime))
	}
}
