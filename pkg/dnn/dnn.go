package dnn

import (
	"fmt"
	"log"

	layer "github.com/artem-97/godnn/pkg/layer"
	loss "github.com/artem-97/godnn/pkg/loss"
)

// DNN -- main struct for dnn
type DNN struct {
	layers []layer.Layer
	loss   loss.Loss
}

// NewDNN is DNN constructor
// returns pointer to DNN
func NewDNN(layers []layer.Layer, loss loss.Loss) *DNN {
	dnn := new(DNN)
	dnn.layers = layers
	dnn.loss = loss
	return dnn
}

// MakeDNN is DNN constructor
// returns DNN
func MakeDNN(layers []layer.Layer, loss loss.Loss) DNN {
	return *NewDNN(layers, loss)
}

// Predict values with DNN
func (dnn *DNN) Predict(xs []float64) ([]float64, error) {
	ys := xs
	for _, layer := range dnn.layers {
		out, err := layer.Forward(ys)
		if err != nil {
			log.Fatalf("Error in forward pass of dnn: %v", err)
			return xs, err
		}
		ys = out
	}
	return ys, nil
}

// Loss calculation for DNN
func (dnn *DNN) Loss(xs []float64, ys []float64) (float64, error) {
	preds, err := dnn.Predict(xs)
	if err != nil {
		log.Fatalf("could not calculate dnn loss: %v", err)
	}
	return dnn.loss.Forward(ys, preds)
}

// Fit sample
func (dnn *DNN) Fit(xs []float64, ys []float64) (float64, error) {
	// forward pass
	var points [][]float64
	newXs := xs
	points = append(points, newXs)
	for _, layer := range dnn.layers {
		out, err := layer.Forward(newXs)
		if err != nil {
			return -1, fmt.Errorf("dnn fit error, %w", err)
		}
		newXs = out
		points = append(points, newXs)
	}
	// calc loss
	point := points[len(points)-1]

	loss, err := dnn.loss.Forward(ys, point)
	// fmt.Printf("epoch : %d,\tloss: %f\n", epoch, loss)
	if err != nil {
		return -1, fmt.Errorf("dnn fit error, %v", err)
	}

	//calc grad
	grad, err := dnn.loss.Backward(ys, point)
	if err != nil {
		return -1, fmt.Errorf("dnn fit error, %w", err)
	}

	point = points[len(points)-2]
	dnn.layers[len(dnn.layers)-1].Backward(point, grad)

	return loss, nil
}
