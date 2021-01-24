# GoDNN
Golang library for playing around with neural networks.


## Installation
```
go get github.com/artem-97/godnn
```


## Example
```go
package main

import (
	"fmt"
	"log"
	"sync"

	"github.com/artem-97/godnn/pkg/activation"
	"github.com/artem-97/godnn/pkg/dataset"
	"github.com/artem-97/godnn/pkg/dnn"
	"github.com/artem-97/godnn/pkg/layer"
	"github.com/artem-97/godnn/pkg/loss"
)


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

	var wg sync.WaitGroup
	for _, batch := range trainset.Data {
		for _, sample := range batch {
			wg.Add(1)
			go func(sample dataset.Sample, loss float64, wg *sync.WaitGroup) {
				defer wg.Done()
				inputs := sample.X
				labels := sample.Y
				currLoss, err := model.Fit(inputs, []float64{labels})
				if err != nil {
					log.Fatalf("train loop failed, %v", err)
				}
				trainLoss += currLoss
			}(sample, trainLoss, &wg)
		}
	}
	validLoss := 0.0
	for _, batch := range testset.Data {
		for _, sample := range batch {
			wg.Add(1)
			go func(sample dataset.Sample, loss float64, wg *sync.WaitGroup) {
				defer wg.Done()
				inputs := sample.X
				labels := sample.Y
				currLoss, err := model.Loss(inputs, []float64{labels})
				if err != nil {
					log.Fatalf("train loop failed, %v", err)
				}
				validLoss += currLoss
			}(sample, validLoss, &wg)
		}
	}
	wg.Wait()
	return trainLoss / float64(trainset.Size), validLoss / float64(testset.Size)
}

func main() {
	var trainset dataset.Dataset
	xTrainFile := "data/X_train.csv"
	yTrainFile := "data/y_train.csv"

	var testset dataset.Dataset
	xTestFile := "data/X_test.csv"
	yTestFile := "data/y_test.csv"

	BatchSize := 1
	trainset.Load(xTrainFile, yTrainFile, BatchSize)
	testset.Load(xTestFile, yTestFile, BatchSize)

	model := createModel()

	Epochs := 100
	// train loop
	for epoch := 0; epoch < Epochs; epoch++ {
		trainLoss, validLoss := trainEpoch(&model, trainset, testset)
		fmt.Printf("epoch: %d, train loss: %f, validation loss: %f\n", epoch, trainLoss, validLoss)
	}
}
```

## Benchmarks with PyTorch
Training loops for same GoDNN and PyTorch models: 
```
training PyTorch model
epoch: 0, train loss: 0.6887095740863255, validation loss : 0.6880764876093183
...
epoch: 99, train loss: 0.19760255834886006, validation loss : 0.20599977672100067
time: 57.93641757965088s


training GoDNN model
epoch: 0, train loss: 0.596076, validation loss: 0.502131
...
epoch: 99, train loss: 0.323762, validation loss: 0.321529
time: 1.486479091s
```

GoDNN model trains about 40 times faster, while having the same performance:
```
GoDNN model
              precision    recall  f1-score   support

         0.0       0.87      0.90      0.88      1684
         1.0       0.89      0.86      0.87      1616

    accuracy                           0.88      3300
   macro avg       0.88      0.88      0.88      3300
weighted avg       0.88      0.88      0.88      3300

PyTorch model
              precision    recall  f1-score   support

         0.0       0.87      0.90      0.89      1684
         1.0       0.89      0.86      0.88      1616

    accuracy                           0.88      3300
   macro avg       0.88      0.88      0.88      3300
weighted avg       0.88      0.88      0.88      3300
```