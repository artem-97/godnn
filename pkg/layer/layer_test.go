package layer

import (
	"fmt"
	"testing"

	"github.com/artem-97/godnn/pkg/activation"
)

// type testpair struct {
// 	ys   []float64
// 	pred []float64
// 	loss float64
// }

// var l2Pairs = []testpair{
// 	testpair{
// 		[]float64{1, 2, 3},
// 		[]float64{1, 2, 3},
// 		0,
// 	},
// 	testpair{
// 		[]float64{2, 2, 3},
// 		[]float64{1, 2, 3},
// 		1,
// 	},
// 	testpair{
// 		[]float64{2, 2, 3, 4},
// 		[]float64{1, 2, 3},
// 		1,
// 	},
// }

func TestForward(t *testing.T) {
	// var sigmoid activation.Sigmoid
	layer := MakeLayer(2, 2, &activation.Sigmoid{})
	a, _ := layer.Forward([]float64{1, 2})
	b, _ := layer.Backward([]float64{1, 2}, []float64{1, 2})
	fmt.Println("b:", b)
	fmt.Println("a:", a)
	// fmt.Println(layer)
	// var loss L2
	// for _, testSample := range l2Pairs {
	// 	predicted, err := loss.forward(testSample.ys, testSample.pred)
	// 	if err != nil {
	// 		t.Errorf("Error: %v", err)
	// 	}
	// 	if predicted != testSample.loss {
	// 		t.Errorf("predicted: %f \t expected: %f", predicted, testSample.loss)
	// 	}

	// }
}
