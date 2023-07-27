import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite


INTERPRETER = None

def loadModel(model_path):

    global INTERPRETER
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    INTERPRETER = tflite.Interpreter(model_path, num_threads=24)
    INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = INTERPRETER.get_input_details()
    output_details = INTERPRETER.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

def embeddings(sample, model_path):

    global INTERPRETER

    # Does interpreter exist?
    if INTERPRETER == None:
        loadModel(model_path)

    # Reshape input tensor
    INTERPRETER.resize_tensor_input(INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
    INTERPRETER.allocate_tensors()

    # Extract feature embeddings
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample, dtype='float32'))
    INTERPRETER.invoke()
    features = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

    return features


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation

def addLinearClassifier(origin_model, num_labels, hidden_units=0):
    
    # Add dense layer with 'relu' activation if hidden_units > 0
    # The output of the origin_model is used as input for this layer
    if hidden_units > 0:
        x = Dense(hidden_units, activation='relu')(origin_model.output)
    else:
        x = origin_model.output
    
    # Add dense layer outputting raw scores for each class
    predictions = Dense(num_labels)(x)
    
    # Transform raw scores to probabilities with sigmoid activation
    output = Activation('sigmoid')(predictions)

    # Now we create the new model
    new_model = Model(inputs=origin_model.input, outputs=output)

    return new_model



def trainNewModel(new_model, train_layers_num, x_train, y_train, x_val, y_val, file_paths_train, file_paths_val, epochs, batch_size, learning_rate, on_epoch_end=None):
    
    tf.keras.backend.clear_session()
    
    # import keras
    from tensorflow import keras

    class FunctionCallback(keras.callbacks.Callback):
        def __init__(self, on_epoch_end=None) -> None:
            super().__init__()
            self.on_epoch_end_fn = on_epoch_end
        
        def on_epoch_end(self, epoch, logs=None):
            if self.on_epoch_end_fn:
                self.on_epoch_end_fn(epoch, logs)
    
    # freeze layer except for train_layers_num
    for layer in new_model.layers[:-train_layers_num]:
        layer.trainable = False
        
# DEBUG        
    print("Length of trainable Weightslen ", len(new_model.trainable_weights))
    
    # Set random seed
    np.random.seed(42)

    # Shuffle train data
    # idx = np.arange(x_train.shape[0])
    # np.random.shuffle(idx)
    # x_train = x_train[idx]
    # y_train = y_train[idx]
    
# DEBUG
    # print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    # print(f"x_train first entries: {x_train[:5]}, y_train first entries: {y_train[:5]}")
    # print(f"unique labels in y_train: {np.unique(y_train).size}")
    # print(f"file_paths_train size: {len(file_paths_train)}, first entries: {file_paths_train[:5]}")
    
    # Shuffle validation data
    # idx = np.arange(x_val.shape[0])
    # np.random.shuffle(idx)
    # x_val = x_val[idx]
    # y_val = y_val[idx]
    
    
# DEBUG
    # print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    # print(f"x_val first entries: {x_val[:5]}, y_val first entries: {y_val[:5]}")
    # print(f"unique labels in y_val: {np.unique(y_val).size}")
    # print(f"file_paths_val size: {len(file_paths_val)}, first entries: {file_paths_val[:5]}")

    # Early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        FunctionCallback(on_epoch_end=on_epoch_end)
    ]
    
    # Cosine annealing lr schedule
    #lr_schedule = keras.experimental.CosineDecay(learning_rate, epochs * x_train.shape[0] / batch_size)

    # Compile model
    new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5), 
                       loss='binary_crossentropy', 
                       metrics=['accuracy', 
                                tf.keras.metrics.Precision(name='prec'), 
                                tf.keras.metrics.Recall(name='recall')])
    
    # Creating generators for raw audio input
    train_gen = AudioDataGenerator(file_paths_train, y_train, batch_size=batch_size)
    val_gen = AudioDataGenerator(file_paths_val, y_val, batch_size=batch_size)

    # Train model
    history = new_model.fit(train_gen, 
                              epochs=epochs, 
                              validation_data=val_gen, 
                              callbacks=callbacks)

    return new_model, history


class AudioDataGenerator(tf.keras.utils.Sequence):
#Use AudioDataGenerator for audiofile batch loading during training

    #Initialize AudioDataGEnerator
    def __init__(self, file_paths, labels, batch_size, samplerate=48000, clip_length=3.0):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.rate = samplerate
        self.clip_length = clip_length

    #Numer of batches per epoch    
    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    #generate databatch
    def __getitem__(self, idx):
        batch_x = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # DEBUG
        # print(f"batch_x size: {len(batch_x)}, first entries: {batch_x[:5]}")
        # print(f"batch_y shape: {batch_y.shape}, first entries: {batch_y[:5]}")

        return np.array([self.load_audio_file(file_name) for file_name in batch_x]), np.array(batch_y)

    #load the audiofile, extract segment
    def load_audio_file(self, file_path):
        import audio
        
         # DEBUG
        # ("Loading audio file: ", file_path) 
    
        sig, rate = audio.openAudioFile(file_path)
        sig = audio.cropCenter(sig, self.rate, self.clip_length)
        
        # DEBUG        
        # ("Loaded audio file, signal shape: ", sig.shape)
    
        return sig
    
    
    
def debug_model(model, file_paths_train, y_train, batch_size):
        
    # Creating generators for raw audio input
    train_gen = AudioDataGenerator(file_paths_train, y_train, batch_size=batch_size) 

    # mini model just until third layer
    mini_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[2].output)

    # intialize list for ID of batch where NaN might happen
    nan_samples = []

    # Going through all batches
    for batch_index in range(len(train_gen)):
        batch_x, batch_y = train_gen[batch_index]
        
        # Inference with batch
        out = mini_model.predict(batch_x)
        
        # Check for NaN value in batch
        if np.isnan(out).any():
            print(f"NaN values found in batch {batch_index}")

            # Inference for every file in current batch
            for i in range(batch_x.shape[0]):
                sample_out = mini_model.predict(batch_x[i:i+1])

                # Check for NaN value for each file
                if np.isnan(sample_out).any():
                        print(f"NaN values found in sample {i} within batch {batch_index}")
                        print(f"File path: {file_paths_train[batch_index * train_gen.batch_size + i]}")
                        nan_samples.append((batch_index, i))
                    
    print(f"Total samples with NaN values: {len(nan_samples)}")

    return nan_samples
