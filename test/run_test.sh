echo "generating dataset\n"
python3 generate_data.py
echo "\n"

echo "training PyTorch model"
python3 train_torch_model.py
echo "\n"

echo "training GoDNN model"
go run godnn_model.go
echo "\n"

echo "classification report"
python3 classification_report.py
