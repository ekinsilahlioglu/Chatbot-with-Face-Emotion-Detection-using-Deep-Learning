"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION

num_classes = 5
img_rows,img_cols = 64,64
batch_size = 32


# parameters

num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 5
patience = 50
train_data_dir = (r'C:\Users\yesil\OneDrive\Masaüstü\face_emotion_detection\datasets\train')
validation_data_dir = (r'C:\Users\yesil\OneDrive\Masaüstü\face_emotion_detection\datasets\validation')



# data generator
train_datagen = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()



#  fer2013_mini_XCEPTION.102-0.66.hdf5

checkpoint = ModelCheckpoint('fer2013_mini_XCEPTION.{epoch:02d}-{val_acc:.2f}.hdf5',
                             monitor='val_loss',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          patience=50)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=50/4,
                              verbose=1)

callbacks = [earlystop,checkpoint,reduce_lr]

nb_train_samples = 24256
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=num_epochs,
                callbacks=callbacks,
                verbose = 1,
                validation_data=validation_generator,
                validation_steps = 1

                )
