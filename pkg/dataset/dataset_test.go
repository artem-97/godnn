package dataset

import (
	"testing"
)

var xfile = "data/X.csv"
var yfile = "data/y.csv"

func TestParseFeatureVector(t *testing.T) {
	parseFeatureVector("12.3")
	// fmt.Println(x)
}

func TestParseFeatureFile(t *testing.T) {
	parseFeatureFile(xfile)
	// fmt.Println(xs)
}

func TestParseTargetFile(t *testing.T) {
	parseTargetFile(yfile)
	// fmt.Println(xs)
}

func TestLoad(t *testing.T) {
	var d Dataset
	d.Load(xfile, yfile, 8)
}
