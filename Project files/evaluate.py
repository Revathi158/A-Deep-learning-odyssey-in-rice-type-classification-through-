from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model=None):
    if model is None:
        model = load_model("model/rice_model.h5")

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'dataset/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
