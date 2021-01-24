package activation

import "math"

// Activation function for dnn
type Activation interface {
	Forward([]float64) []float64
	Backward([]float64) []float64
}

// Identity is identical activation function
type Identity struct{}

// Forward pass for inputs
// returns inputs
func (activation *Identity) Forward(xs []float64) []float64 {
	return xs
}

// Backward pass for inputs
// returns [1,1,1,...,1]
func (activation *Identity) Backward(xs []float64) []float64 {
	for i := range xs {
		xs[i] = 1
	}
	return xs
}

// Sigmoid activation function
type Sigmoid struct{}

// Forward pass for inputs
// returns elementwise sigmoid(inputs) = outputs
// sigma [x] = 1 / (1 + exp(-x))
func (activation *Sigmoid) Forward(xs []float64) []float64 {
	for i := 0; i < len(xs); i++ {
		xs[i] = 1 / (1 + math.Exp(-xs[i]))
	}
	return xs
}

// Backward pass for inputs
// returns elementwise [gradient of sigmoid](inputs) = outputs
// dsigma [x] = sigma * (1-sigma)[x]
func (activation *Sigmoid) Backward(xs []float64) []float64 {
	for i := 0; i < len(xs); i++ {
		xs[i] = (1 / (1 + math.Exp(-xs[i]))) * (1 - 1/(1+math.Exp(-xs[i])))
	}
	return xs
}

// ReLU activation function
type ReLU struct{}

// Forward pass for inputs
// relu [x] = x if x >=0, else 0
func (activation *ReLU) Forward(xs []float64) []float64 {
	for i := 0; i < len(xs); i++ {
		if xs[i] < 0 {
			xs[i] = 0
		}
	}
	return xs
}

// Backward pass for inputs
// returns elementwise [gradient of sigmoid](inputs) = outputs
// drelu [x] = 1 if x>=0, else 0
func (activation *ReLU) Backward(xs []float64) []float64 {
	for i := 0; i < len(xs); i++ {
		if xs[i] >= 0 {
			xs[i] = 1
		} else {
			xs[i] = 0
		}
	}
	return xs
}
