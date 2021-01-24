package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"time"

	"github.com/artem-97/godnn/pkg/activation"
	"github.com/artem-97/godnn/pkg/dataset"
	"github.com/artem-97/godnn/pkg/dnn"
	"github.com/artem-97/godnn/pkg/layer"
	"github.com/artem-97/godnn/pkg/loss"
)

type params struct {
	BatchSize    int     `json:"batch_size"`
	Epochs       int     `json:"epochs"`
	LearningRate float64 `json:"learning_rate"`
}

func readParams() params {
	paramsFile, err := os.Open("params.json")
	if err != nil {
		log.Fatalf("could not read params file: %v", err)
	}
	defer paramsFile.Close()

	byteValue, _ := ioutil.ReadAll(paramsFile)

	var result params
	json.Unmarshal(byteValue, &result)
	return result

}

func createModel() dnn.DNN {
	dIn := 2
	dOut := 1

	dH1 := 5
	dH2 := 3

	fc1 := layer.MakeLayer(dIn, dH1, &activation.ReLU{})
	fc2 := layer.MakeLayer(dH1, dH2, &activation.ReLU{})
	fc3 := layer.MakeLayer(dH2, dOut, &activation.Sigmoid{}) // BCEWithLogitsLoss

	model := dnn.MakeDNN([]layer.Layer{fc1, fc2, fc3}, &loss.LogLoss{})
	return model
}

// returns (train loss, validation loss)
func trainEpoch(model *dnn.DNN, trainset dataset.Dataset, testset dataset.Dataset) (float64, float64) {
	trainLoss := 0.0

	for _, batch := range trainset.Data {
		// fmt.Println("batch: ", batchNum)
		for _, sample := range batch {
			inputs := sample.X
			labels := sample.Y
			currLoss, err := model.Fit(inputs, []float64{labels})
			if err != nil {
				log.Fatalf("train loop failed, %v", err)
			}
			trainLoss += currLoss
		}
	}

	validLoss := 0.0
	for _, batch := range testset.Data {
		// fmt.Println("batch: ", batchNum)
		for _, sample := range batch {
			inputs := sample.X
			labels := sample.Y
			currLoss, err := model.Loss(inputs, []float64{labels})
			if err != nil {
				log.Fatalf("train loop failed, %v", err)
			}
			validLoss += currLoss
		}
	}
	return trainLoss / float64(trainset.Size), validLoss / float64(testset.Size)
}

func dumpPreds(preds []float64) {
	predStr := strings.Trim(strings.Join(strings.Fields(fmt.Sprint(preds)), "\n"), "[]")

	err := ioutil.WriteFile("data/godnn_pred.csv", []byte(predStr), 0644)
	if err != nil {
		log.Fatalf("could not dump predictions of model: %v", err)
	}
}

func main() {
	trainParams := readParams()

	var trainset dataset.Dataset
	xTrainFile := "data/X_train.csv"
	yTrainFile := "data/y_train.csv"

	var testset dataset.Dataset
	xTestFile := "data/X_test.csv"
	yTestFile := "data/y_test.csv"

	// trainset.Load(xTrainFile, yTrainFile, trainParams.BatchSize)
	BatchSize := 1
	trainset.Load(xTrainFile, yTrainFile, BatchSize)
	testset.Load(xTestFile, yTestFile, BatchSize)

	model := createModel()

	// train loop
	start := time.Now()
	for epoch := 0; epoch < trainParams.Epochs; epoch++ {
		trainLoss, validLoss := trainEpoch(&model, trainset, testset)
		fmt.Printf("epoch: %d, train loss: %f, validation loss: %f\n", epoch, trainLoss, validLoss)
	}
	end := time.Now()
	elapsed := end.Sub(start)
	fmt.Printf("time: %v\n", elapsed)

	// dump predictions
	var preds []float64
	for _, batch := range testset.Data {
		for _, sample := range batch {
			pred, err := model.Predict(sample.X)
			if err != nil {
				log.Fatalf("could not predict sample %v", sample)
			}
			if pred[0] > 0.5 {
				preds = append(preds, 1)
			} else {
				preds = append(preds, 0)
			}
		}
	}
	dumpPreds(preds)
}
