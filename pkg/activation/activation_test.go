package activation

import (
	"fmt"
	"testing"
)

type testpair struct {
	inputs  []float64
	outputs []float64
}

func TestParseFeatureVector(t *testing.T) {
	xs := []float64{1, -2, -1, 0}
	s := Sigmoid{}
	fmt.Println(s.Forward(xs))
	fmt.Println(s.Backward(xs))
}
