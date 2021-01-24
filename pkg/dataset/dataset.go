package dataset

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

// Sample from dataset {x : fuature_vector, y: target_val}
type Sample struct {
	X []float64
	Y float64
}

// Dataset structure, constructs from
// xfile and yfile
type Dataset struct {
	Xfile     string
	Yfile     string
	BatchSize int
	Size      int
	Data      [][]Sample
}

func parseFeatureVector(features string) ([]float64, error) {
	var result []float64

	delimiter := ","
	featuresStr := strings.Split(features, delimiter)
	for _, featureStr := range featuresStr {
		featureFloat, err := strconv.ParseFloat(featureStr, 32)
		if err != nil {
			log.Fatalf("Could not parse feature: %v, error: %v", featureStr, err)
			return result, nil
		}
		result = append(result, featureFloat)
	}
	return result, nil
}

func parseFeatureFile(xfile string) ([][]float64, error) {
	var xs [][]float64

	xf, err := os.Open(xfile)
	if err != nil {
		log.Fatalf("Could not read feature file: %v, error: %v", xfile, err)
	}
	defer xf.Close()

	scanner := bufio.NewScanner(xf)
	for scanner.Scan() {

		xstr := scanner.Text()
		x, err := parseFeatureVector(xstr)
		if err != nil {
			log.Fatalf("%v", err)
		}
		xs = append(xs, x)
	}
	return xs, nil
}

func parseTargetFile(yfile string) ([]float64, error) {

	var ys []float64

	yf, err := os.Open(yfile)
	if err != nil {
		log.Fatalf("Could not read target file: %v, error: %v", yfile, err)
	}
	defer yf.Close()

	scanner := bufio.NewScanner(yf)
	for scanner.Scan() {

		ystr := scanner.Text()
		y, err := strconv.ParseFloat(ystr, 32)
		if err != nil {
			log.Fatalf("could not convert target: %v to float, error: %v", ystr, err)
		}
		ys = append(ys, y)
	}
	return ys, nil
}

// Load dataset from file
func (dataset *Dataset) Load(xfile string, yfile string, batchSize int) error {
	dataset.Xfile = xfile
	dataset.Yfile = yfile
	dataset.BatchSize = batchSize

	xs, err := parseFeatureFile(xfile)
	if err != nil {
		return fmt.Errorf("could not load dataset, %v", err)
	}
	ys, err := parseTargetFile(yfile)
	if err != nil {
		return fmt.Errorf("could not load dataset, %v", err)
	}

	if len(xs) != len(ys) {
		return fmt.Errorf("different length of train{%d} and test{%d} data", len(xs), len(ys))
	}
	dataset.Size = len(xs)

	for batch := 0; batch < len(xs)/batchSize+1; batch++ {
		var newBatch []Sample
		// fmt.Printf("batch:%v\n", batch)
		for i := batch * batchSize; i < (batch+1)*batchSize; i++ {
			if i < len(xs) {
				// fmt.Printf("i:%v\n", i)
				newBatch = append(newBatch, Sample{xs[i], ys[i]})
			}
			// fmt.Println(len(newBatch))
		}
		dataset.Data = append(dataset.Data, newBatch)
	}
	return nil
}
