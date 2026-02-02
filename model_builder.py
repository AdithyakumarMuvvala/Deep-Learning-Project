from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate, Dropout

# Updated for RGB
IMG_SHAPE = (128, 128, 3)
NUM_CLASSES = 4

def build_baseline_model():
    """
    Standard sequential CNN for Multi-Class.
    """
    model = Sequential([
        Input(shape=IMG_SHAPE),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        # Output layer for 4 classes
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def build_multiscale_model():
    """
    Multi-scale CNN for Multi-Class.
    """
    input_layer = Input(shape=IMG_SHAPE)
    
    # Branch 1
    b1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    b1 = MaxPooling2D((2, 2))(b1)
    
    # Branch 2 (Dilated)
    b2 = Conv2D(32, (3, 3), padding='same', dilation_rate=2, activation='relu')(input_layer)
    b2 = MaxPooling2D((2, 2))(b2)
    
    # Branch 3 (Large Kernel)
    b3 = Conv2D(32, (5, 5), padding='same', activation='relu')(input_layer)
    b3 = MaxPooling2D((2, 2))(b3)
    
    merged = concatenate([b1, b2, b3])
    
    x = Conv2D(64, (3, 3), activation='relu')(merged)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer for 4 classes
    output_layer = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
