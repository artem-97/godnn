package loss

import (
	"fmt"
	"testing"
)

type testpair struct {
	yTrue []float64
	yPred []float64
	loss  float64
	err   error
}

var l2Pairs = []testpair{
	{
		[]float64{1, 2, 3},
		[]float64{1, 2, 3},
		0,
		nil,
	},
	{
		[]float64{2, 2, 3},
		[]float64{1, 2, 3},
		1,
		nil,
	},
	{
		[]float64{2, 2, 3, 4},
		[]float64{1, 2, 3},
		1,
		fmt.Errorf("loss dimension error: len(yTrue) = 4, len(yPred) = 3"),
	},
}

func TestL2(t *testing.T) {
	var loss L2
	for _, testSample := range l2Pairs {
		predicted, err := loss.Forward(testSample.yTrue, testSample.yPred)
		if err != nil {
			if err.Error() != testSample.err.Error() {
				t.Errorf("predicted error: %v \t expected error: %v", testSample.err, err)
			}
			continue
		}
		if predicted != testSample.loss {
			t.Errorf("predicted: %f \t expected: %f", predicted, testSample.loss)
		}

	}
}
