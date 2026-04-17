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

### TODO
-   [x] Develop a base test script
-   [ ] Train the model with acceptable result
-   [ ] Create a report
-   [ ] Write a docker image
