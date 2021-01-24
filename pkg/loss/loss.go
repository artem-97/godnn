package loss

import (
	"fmt"
	"math"
)

// Loss function for dnn
type Loss interface {
	Forward([]float64, []float64) (float64, error)
	Backward([]float64, []float64) ([]float64, error)
}

// L2 loss function
type L2 struct{}

// Forward pass for loss
// returns loss and error if idxs mismatch
func (lossL2 *L2) Forward(yTrue []float64, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, fmt.Errorf("loss dimension error: len(yTrue) = %d, len(yPred) = %d", len(yTrue), len(yPred))
	}

	loss := 0.0
	for i := 0; i < len(yPred); i++ {
		loss += (yPred[i] - yTrue[i]) * (yPred[i] - yTrue[i])
	}
	return loss, nil

}

// Backward pass for loss
// returns gradient vector and error if idxs mismatch
func (lossL2 *L2) Backward(yTrue []float64, yPred []float64) ([]float64, error) {
	if len(yTrue) != len(yPred) {
		return yTrue, fmt.Errorf("loss dimension error: len(yTrue) = %d, len(yPred) = %d", len(yTrue), len(yPred))
	}

	grad := make([]float64, len(yTrue))
	for i := 0; i < len(yPred); i++ {
		grad[i] = yPred[i] - yTrue[i]
	}
	return grad, nil

}

// LogLoss loss function
type LogLoss struct{}

// Forward pass for loss
// returns loss and error if idxs mismatch
func (lossLL *LogLoss) Forward(yTrue []float64, yPred []float64) (float64, error) {
	if len(yTrue) != len(yPred) {
		return 0, fmt.Errorf("loss dimension error: len(yTrue) = %d, len(yPred) = %d", len(yTrue), len(yPred))
	}

	loss := 0.0
	for i := 0; i < len(yPred); i++ {
		if yPred[i] == 1 {
			continue
		}
		if yPred[i] == 0 {
			continue
		}
		loss -= yTrue[i] * math.Log(yPred[i])
		loss -= (1 - yTrue[i]) * math.Log((1 - yPred[i]))
	}
	return loss, nil

}

// Backward pass for loss
// returns gradient vector and error if idxs mismatch
func (lossLL *LogLoss) Backward(yTrue []float64, yPred []float64) ([]float64, error) {
	if len(yTrue) != len(yPred) {
		return yTrue, fmt.Errorf("loss dimension error: len(yTrue) = %d, len(yPred) = %d", len(yTrue), len(yPred))
	}

	grad := make([]float64, len(yTrue))
	for i := 0; i < len(yPred); i++ {
		grad[i] -= yTrue[i] / yPred[i]
		grad[i] += (1 - yTrue[i]) / (1 - yPred[i])
	}
	return grad, nil

}
