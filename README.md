## Format
-   filename: "<model>_cloud.ipynb"
-   `model_path`: "best_<model>.keras"

### Change the model here
```python
base_model = keras.applications.MobileNetV2( # Change MobileNetV2 to something else
    include_top=False, 
    weights='imagenet', 
    input_shape=input_shape
)
base_model.trainable = True

# ------------------------------------------------------------------------------------

inputs = keras.Input(shape=input_shape)
x = keras.applications.mobilenet_v2.preprocess_input(inputs) # Change here too
x = base_model(x, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
```

### TODO
-   [ ] Develop a base test script
-   [ ] Train the model with acceptable result
-   [ ] Create a report
-   [ ] Write a docker image
