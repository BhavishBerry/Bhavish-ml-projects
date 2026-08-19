"""Train the cat/dog CNN (same architecture as cnn_project.ipynb) and save model.h5
into each theme folder of the Streamlit app, so it's plug-and-play for everyone."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

BASE = Path(__file__).resolve().parent.parent / "Bhavish" / "cnn_project"
tr_location = str(BASE / "catdog_real_dataset_1" / "train")
val_location = str(BASE / "catdog_real_dataset_1" / "validation")

train = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

train_data = train.flow_from_directory(tr_location, target_size=(64, 64), batch_size=8, class_mode="binary")
valid_data = validation.flow_from_directory(val_location, target_size=(64, 64), batch_size=8, class_mode="binary")
print("class_indices:", train_data.class_indices)

model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer=RMSprop(learning_rate=0.001), metrics=["accuracy"])
history = model.fit(train_data, epochs=10, validation_data=valid_data)

webapp = BASE / "catdog_webapp"
for theme in ["theme1", "theme2", "theme3"]:
    out = webapp / theme / "model.h5"
    model.save(out)
    print("saved", out)
