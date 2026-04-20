## Format
-   Python Version: `3.13.11`

### Training
```bash
python3 trainer.py
```

### Change the model here
```python
# Model Initialize
model_base = MobileNetMultiHead(num_classes=num_classes)
```
_Add new model in the `model.py`_

### Prediction
```bash
python3 predicter.py
# it will generate 'predictions.jpg'
```

### Host an app
```bash
# With docker
docker build -t deep_app -f Dockerfile.app .
docker run -p 8501:8501 deep_app

# Local
streamlit run app.py
```

### TODO
-   [x] Develop a base test script
-   [ ] Train the model with acceptable result
-   [ ] Create a report
-   [ ] Write a docker image
