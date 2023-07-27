import os
import sys
import argparse

import numpy as np

import audio
import model

import tensorflow as tf

def loadData(data_path, model_path):

    # Get list of subfolders as labels
    labels = [l for l in sorted(os.listdir(data_path))]

    # Load data
    x_emb = []
    y_emb = []
    file_paths = []
    
    for i, label in enumerate(labels):
            
            # Get label vector            
            label_vector = np.zeros((len(labels),), dtype='float32')
            if not label.lower() in ['noise', 'other', 'background', 'silence']:
                label_vector[i] = 1
    
            # Get list of files
            files = [os.path.join(data_path, label, f) for f in sorted(os.listdir(os.path.join(data_path, label))) if f.rsplit('.', 1)[1].lower() in ['wav', 'flac', 'mp3', 'ogg', 'm4a']]

            # Load files
            for j, f in enumerate(files):
    
                # removed all actions for emebedding for faster load
        
                # Load audio
                # sig, rate = audio.openAudioFile(f)
    
                # Crop center segment (3.0 s)
                # sig = audio.cropCenter(sig, rate, 3.0)

                # Get feature embeddings
                embeddings = 0
                # embeddings = model.embeddings([sig], model_path)[0]

                # Add to training data
                x_emb.append(embeddings)
                y_emb.append(label_vector)
                
                # Add the file path
                file_paths.append(f)
                
                # Print info every 100 iterations
                if j % 100 == 0:
                    print(f'Processed {j} files. Currently processing file: {f}')

    # Convert to numpy arrays
    x_emb = np.array(x_emb, dtype='float32')
    y_emb = np.array(y_emb, dtype='float32')
    file_paths = np.array(file_paths)

    return x_emb, y_emb, labels, file_paths