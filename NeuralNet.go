package main

import (
	"fmt"
	mat "github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
)

func matRandomInitVec(X *mat.Vector) {
	r, _ := X.Dims()
	for i := 0; i < r; i++ {
		X.SetVec(i, rand.Float64())
	}
}
func matRandomInitMat(X *mat.Dense) {
	r, c := X.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			X.Set(i, j, rand.Float64())
		}
	}
}

func Transpose(X *mat.Dense, Y *mat.Dense) {
	_, c := X.Dims()
	yr, _ := Y.Dims()
	if c == yr {
		for i := 0; i < yr; i++ {
			X.Set(0, i, Y.At(i, 0))
		}
	} else {
		fmt.Println("ERROR: Transpose fails ")
	}
}
func matPrint(name string, X mat.Matrix) {
	return
	r, c := X.Dims()
	fmt.Println(name, " rows: ", r, " cols: ", c)
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
func mapActivation(X *mat.Dense, col int) {
	r, _ := X.Dims()
	for i := 0; i < r; i++ {
		if nn.activation_function == "sigmoid" {
			sigmoid := 1 / (1 + math.Exp(-X.At(i, 0)))
			X.Set(i, col, sigmoid)
		} else {
			X.Set(i, col, math.Tanh(X.At(i, 0)))
		}
	}
}
func mapDeActivation(X *mat.Dense) {
	r, _ := X.Dims()
	for i := 0; i < r; i++ {
		y := X.At(i, 0)
		if nn.activation_function == "sigmoid" {
			sigmoid := y * (1 - y)
			X.Set(i, 0, sigmoid)
		} else {
			X.Set(i, 0, (1 - (y * y)))
		}
	}
}
/*func vectorMultiplyScaler(X *mat.Dense, y float64) {
	rx, _ := X.Dims()

	for i := 0; i < rx; i++ {
		x := X.At(i, 0)
		X.Set(i, 0, x*y)
	}
}*/
func vectorAdd(X *mat.Dense, Y *mat.Dense, xcol int) {
	rx, _ := X.Dims()
	ry, _ := Y.Dims()
	if rx != ry {
		fmt.Println("ERROR in multiple vector")
	}
	for i := 0; i < rx; i++ {
		y := Y.At(i, 0)
		x := X.At(i, xcol)
		X.Set(i, xcol, x+y)
	}
}
func vectorsMultiply(X *mat.Dense, Y *mat.Dense, learning_rate float64) {
	rx, _ := X.Dims()
	ry, _ := Y.Dims()
	if rx != ry {
		fmt.Println("ERROR in multiple vector")
	}
	for i := 0; i < rx; i++ {
		y := Y.At(i, 0)
		x := X.At(i, 0)
		X.Set(i, 0, x*y*learning_rate)
	}
}

type LayerType struct {
	nodes                           int
	weights, weights_delta, output_T *mat.Dense
	error_val, output, bias        *mat.Dense // output genereated at each layer, inout is special case.
	gradients                       *mat.Dense
}

type NeuralNet struct {
	input_T        *mat.Dense
	layers         []LayerType
	total_layers   int
	max_batch_size int

	learning_rate       float64
	errorval            float64
	activation_function string
	debug               bool
}

func (n *NeuralNet) Init(layer_count []int) {
	total_layers := len(layer_count)
	nn.total_layers = total_layers

	n.learning_rate = 0.04
	n.activation_function = "sigmoid"
	n.layers = make([]LayerType, total_layers)
	for i := 0; i < total_layers; i++ {
		n.layers[i].nodes = layer_count[i]
	}
	n.max_batch_size = 10
	n.layers[0].output = mat.NewDense(n.layers[0].nodes, n.max_batch_size, make([]float64, n.max_batch_size*n.layers[0].nodes))
	n.layers[0].output_T = mat.NewDense(1, n.layers[0].nodes, make([]float64, 1*n.layers[0].nodes))
	for i := 1; i < total_layers; i++ {
		curr_nodes := n.layers[i].nodes
		prev_nodes := n.layers[i-1].nodes

		n.layers[i].weights = mat.NewDense(curr_nodes, prev_nodes, make([]float64, curr_nodes*prev_nodes))
		n.layers[i].weights_delta = mat.NewDense(curr_nodes, prev_nodes, make([]float64, curr_nodes*prev_nodes))
		n.layers[i].output_T = mat.NewDense(1, curr_nodes, make([]float64, 1*curr_nodes))
		matRandomInitMat(n.layers[i].weights)

		n.layers[i].output = mat.NewDense(curr_nodes, 1, make([]float64, 1*curr_nodes))
		n.layers[i].error_val = mat.NewDense(curr_nodes, 1, make([]float64, 1*curr_nodes))

		n.layers[i].bias = mat.NewDense(curr_nodes, 1, make([]float64, 1*curr_nodes))
		n.layers[i].gradients = mat.NewDense(curr_nodes, 1, make([]float64, 1*curr_nodes))
		matRandomInitMat(n.layers[i].bias)
	}
	n.debug = true
}
func predict(input_args []float64, target_args []float64, batch_size int) {
	last_layer := nn.total_layers - 1
	targets := mat.NewDense(nn.layers[last_layer].nodes, 1, make([]float64, 1*nn.layers[last_layer].nodes))
	targets.SetCol(0, target_args)

	/* FORWARD PROPAGATION 1 to last layer : Generating output using input */
	//batch_size := 1
	for b := 0; b < batch_size; b++ {
		nn.layers[0].output.SetCol(b, input_args)
	}
	for i := 1; i <= last_layer; i++ {
		for b := 0; b < batch_size; b++ {
			nn.layers[i].output.Mul(nn.layers[i].weights, nn.layers[i-1].output.ColView(b))
			vectorAdd(nn.layers[i].output, nn.layers[i].bias , b)
			mapActivation(nn.layers[i].output,b)
		}
	}
	nn.layers[last_layer].error_val.Sub(targets, nn.layers[last_layer].output)
	nn.errorval = nn.layers[last_layer].error_val.At(0, 0)
	nn.layers[0].output_T.Copy(nn.layers[0].output.T())
}
func train(input_args []float64, target_args []float64, batch_size int) {
	predict(input_args, target_args,batch_size)
	last_layer := nn.total_layers - 1

	if nn.debug {
		fmt.Println(" Input:", input_args, "Target:", target_args, " : ", nn.layers[last_layer].output.At(0, 0))
	}

	/*   BACK PROPOGATION:  weights correction using error
	delte_weight =  learningrate * error * deactivation(x)  : here x is input to that layer not actual input
	delete_bias = learningrate * error
	*/
	for i := last_layer; i > 0; i-- {
		nn.layers[i].gradients.Copy(nn.layers[i].output)
		mapDeActivation(nn.layers[i].gradients)
		vectorsMultiply(nn.layers[i].gradients, nn.layers[i].error_val,nn.learning_rate)

		Transpose(nn.layers[i-1].output_T, nn.layers[i-1].output)
		nn.layers[i].weights_delta.Mul(nn.layers[i].gradients, nn.layers[i-1].output_T)
		nn.layers[i].weights.Add(nn.layers[i].weights, nn.layers[i].weights_delta)
		nn.layers[i].bias.Add(nn.layers[i].bias, nn.layers[i].gradients)

		if (i - 1) > 0 {
			//nn.layers[i-1].error_val.MulVec(nn.layers[i].weights.T(), nn.layers[i].error_val)
			nn.layers[i-1].error_val.Mul(nn.layers[i].weights.T(), nn.layers[i].error_val)
		}
	}
}
