package layer

import (
	"fmt"
	"math/rand"

	"github.com/artem-97/godnn/pkg/activation"
)

// Layer struct for dnn
// dimIn -- input dims
// dimOuy -- output dims
// weights -- dimOut x dimIn matrix
// activation -- activation interface
type Layer struct {
	dimIn      int
	dimOut     int
	weights    [][]float64
	activation activation.Activation
}

// NewLayer is Layer constructor
// returns pointer to Layer
func NewLayer(dimIn int, dimOut int, activation activation.Activation) *Layer {
	layer := new(Layer)
	layer.dimIn = dimIn
	layer.dimOut = dimOut

	// initVal := 0.05

	weights := make([][]float64, dimOut)
	for i := range weights {
		weights[i] = make([]float64, dimIn)
		for j := range weights[i] {
			// weights[i][j] = initVal
			weights[i][j] = rand.Float64()

		}
	}

	layer.weights = weights
	layer.activation = activation

	return layer
}

// MakeLayer is Layer constructor
// returns Layer
func MakeLayer(dimIn int, dimOut int, activation activation.Activation) Layer {
	return *NewLayer(dimIn, dimOut, activation)
}

// Forward pass for Layer
// returns layer.activation(layer.weights @ xs) and error if idxs mismatch
// affine transformation + nonlinear activation
func (layer *Layer) Forward(xs []float64) ([]float64, error) {
	if len(xs) != layer.dimIn {
		return xs, fmt.Errorf("dimension error: Layer dims [%d, %d], input vector dim [%d]", layer.dimIn, layer.dimOut, len(xs))
	}

	ys := make([]float64, int(layer.dimOut))

	for i := 0; i < int(layer.dimOut); i++ {
		acc := 0.0
		for j := 0; j < int(layer.dimIn); j++ {
			acc += layer.weights[i][j] * xs[j]
		}
		ys[i] = acc
	}
	return layer.activation.Forward(ys), nil
}

// Backward pass for layer
// updates layer weights
func (layer *Layer) Backward(xs []float64, grad []float64) ([]float64, error) {
	eta := 0.05
	if len(xs) != layer.dimIn {
		return xs, fmt.Errorf("Dimension error: Layer dims [%d, %d], input vector dim [%d]", layer.dimIn, layer.dimOut, len(xs))
	}
	updWeights := make([]float64, layer.dimOut)
	for i := 0; i < int(layer.dimOut); i++ {
		acc := 0.0
		for j := 0; j < int(layer.dimIn); j++ {
			acc += layer.weights[i][j] * xs[j]
		}
		updWeights[i] = acc
	}

	updWeights = layer.activation.Backward(updWeights)
	for i := 0; i < int(layer.dimOut); i++ {
		for j := 0; j < int(layer.dimIn); j++ {
			layer.weights[i][j] -= eta * xs[j] * grad[i] * updWeights[i]
		}
	}
	newGrad := make([]float64, layer.dimIn)
	for j := 0; j < int(layer.dimIn); j++ {
		for i := 0; i < int(layer.dimOut); i++ {
			newGrad[j] = grad[i] * layer.weights[i][j]
		}
	}

	return newGrad, nil

}
