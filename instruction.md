## Prerequisite
1.  Download the latest <a href="https://universe.roboflow.com/wrkspc-gi0hz/cloud-classification-mf91q/dataset/7">dataset</a> version (2024-12-22 7:33pm) with "Tensorflow Object Detection" format.
2.  Put in the directory as following:
    ```
    Cloud-Classification-7
    ├── README.dataset.txt
    ├── README.roboflow.txt
    ├── test
    ├── train
    └── valid
    ```

### Local
1.  Create a python version 3.13.11 environment.
2.  Install the require python libraries in the `requirements.txt`. </br>
    Conveniently you can run `pip3 install -r requirements.txt`

### Docker
1.  Build the docker image with provided Dockerfile.
    a.  For hosting an streamlit app:
        `docker build -t deep_app -f Dockerfile.app .`
    b.  For achieving develop environment like:
        `docker build -t deep_dev -f Dockerfile.dev_env .`
2.  Run docker image:
    a.  For hosting an streamlit app:
        _Make sure that `model_pth` is exists with proper `.pth`, and the model is set to correct one (see change the model part)_
        `docker run -p 8501:8501 deep_app`
    b.  For testing interactive develop environment:
        `docker run -it deep_dev bash`

# Running source code

### Change the model 
_This part is critical, especially when hosting an app. Make sure to check this part in `app.py`._
```python
# Model Initialize
model_base = MobileNetMultiHead(num_classes=num_classes)
```
_Available model can be found in the `model.py`_

## Training
```bash
python3 trainer.py
```

## Prediction
```bash
python3 predicter.py
# it will generate 'predictions.jpg'
```

## Host an app
```bash
# With docker
docker build -t deep_app -f Dockerfile.app .
docker run -p 8501:8501 deep_app

# Local
streamlit 
