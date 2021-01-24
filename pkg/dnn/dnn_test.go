package dnn

import (
	"fmt"
	"log"
	"testing"

	"github.com/artem-97/godnn/pkg/activation"
	"github.com/artem-97/godnn/pkg/layer"
	"github.com/artem-97/godnn/pkg/loss"
)

type testpair struct {
	inputs  []float64
	outputs []float64
}

func TestDNN(t *testing.T) {
	layer1 := layer.MakeLayer(2, 3, &activation.Sigmoid{})
	layer2 := layer.MakeLayer(3, 3, &activation.Sigmoid{})
	dnn := NewDNN([]layer.Layer{layer1, layer2}, &loss.LogLoss{})
	xs := []float64{1, 2}
	ys := []float64{0, 0, 0}
	for epoch := 0; epoch < 1000; epoch++ {
		_, err := dnn.Fit(xs, ys)
		if err != nil {
			log.Fatalf("%v", err)
		}
	}
	// fmt.Println(o)
	// fmt.Println(dnn)
	o, err := dnn.Predict(xs)
	if err != nil {
		log.Fatalf("%v", err)
	}
	fmt.Println(o)
}
