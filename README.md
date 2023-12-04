# Celebrity Image Classification
### Summary of the Chosen Model, Training Process, and Critical Findings

## Chosen Model
The chosen model is a Convolutional Neural Network (CNN) designed for image classification. The architecture includes convolutional layers for feature extraction, max-pooling layers for spatial downsampling, a flatten layer to convert 2D feature maps to a vector, and dense layers for classification. The model utilizes the softmax activation function in the output layer for multi-class classification.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
```

## Training Process
- The dataset consists of cropped images of celebrities, including Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.
- The dataset is split into training and testing sets, and preprocessing techniques such as resizing and normalization are applied.
- The model is compiled using the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the evaluation metric.
- Training is performed for 30 epochs with a batch size of 128 and a validation split of 10%.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.1)
```

## Critical Findings
- The model achieved an accuracy of approximately 92.5% on the test set.
- The classification report provides a detailed breakdown of precision, recall, and F1-score for each class, offering insights into the model's performance on individual categories.
- The `make_prediction` function enables predictions on new images, enhancing the model's practical utility.

