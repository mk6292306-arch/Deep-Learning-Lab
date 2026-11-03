import tensorflow as tf
from tensorflow.keras import applications, layers, models, optimizers, datasets
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class PretrainedImageClassifier:
    def __init__(self, model_name='MobileNetV2', img_size=(32, 32), num_classes=10):
        """
        Initialize the pre-trained image classifier
        """
        self.model_name = model_name
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, trainable_layers=5, dropout_rate=0.3):
        """
        Build the model with pre-trained base and custom classifier
        """
        # Map model names to their respective functions
        model_dict = {
            'VGG16': applications.VGG16,
            'ResNet50': applications.ResNet50,
            'MobileNetV2': applications.MobileNetV2,
            'EfficientNetB0': applications.EfficientNetB0,
        }
        
        if self.model_name not in model_dict:
            raise ValueError(f"Model {self.model_name} not supported. Choose from {list(model_dict.keys())}")
        
        # Load pre-trained model without top classification layers
        base_model = model_dict[self.model_name](
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Freeze the base model layers initially
        base_model.trainable = False
        
        # Create the complete model
        inputs = layers.Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Preprocessing - simple rescaling for all models
        x = layers.Rescaling(1./255)(inputs)
        
        # Pass through base model
        x = base_model(x, training=False)
        
        # Add custom classifier on top
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        
        # Unfreeze the top n layers for fine-tuning
        if trainable_layers > 0:
            base_model.trainable = True
            # Fine-tune from this layer onwards
            fine_tune_at = len(base_model.layers) - trainable_layers
            
            # Freeze all the layers before the fine_tune_at layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            print(f"Fine-tuning {trainable_layers} layers")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate settings"""
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']  # Only using accuracy to avoid metric errors
        )
    
    def prepare_data(self):
        """Prepare CIFAR-10 dataset for training"""
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        
        # Normalize pixel values to be between 0 and 1
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        # Convert labels to integers
        y_train = y_train.astype('int32').flatten()
        y_test = y_test.astype('int32').flatten()
        
        # Split training data into train and validation sets
        split = int(0.8 * len(x_train))
        x_val = x_train[split:]
        y_val = y_train[split:]
        x_train = x_train[:split]
        y_train = y_train[:split]
        
        # Resize images to match the expected input size of the model
        if self.img_size != (32, 32):
            x_train = tf.image.resize(x_train, self.img_size)
            x_val = tf.image.resize(x_val, self.img_size)
            x_test = tf.image.resize(x_test, self.img_size)
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def train(self, x_train, y_train, x_val, y_val, epochs=5, batch_size=32):
        """Train the model"""
        callbacks = [
            EarlyStopping(patience=2, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=1)
        ]
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, x_test, y_test):
        """Evaluate the model on test data"""
        results = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]*100:.2f}%")
        
        # Make predictions
        predictions = self.model.predict(x_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate additional metrics
        print("\nClassification Report:")
        print(classification_report(y_test, predicted_classes))
        
        return results
    
    def predict_single(self, image):
        """Predict class for a single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_class, confidence
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage with CIFAR-10 dataset
if _name_ == "_main_":
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create classifier
    classifier = PretrainedImageClassifier(
        model_name='MobileNetV2',
        img_size=(32, 32),
        num_classes=10
    )
    
    # Build model
    print("Building model...")
    model = classifier.build_model(trainable_layers=5, dropout_rate=0.3)
    
    # Compile model
    print("Compiling model...")
    classifier.compile_model(learning_rate=0.001)
    
    # Prepare data
    print("Preparing data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = classifier.prepare_data()
    
    print(f"Training data shape: {x_train.shape}")

    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Train model
    print("Training model...")
    history = classifier.train(x_train, y_train, x_val, y_val, epochs=5, batch_size=32)
    
    # Evaluate on test set
    print("Evaluating model...")
    test_results = classifier.evaluate(x_test, y_test)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save the model
    model.save('cifar10_classifier.h5')
    print("Model saved as 'cifar10_classifier.h5'")
    
    # Example prediction on a single image
    sample_idx = 0  # Change this to test different images
    sample_image = x_test[sample_idx]
    true_label = y_test[sample_idx]
    
    predicted_class, confidence = classifier.predict_single(sample_image)
    
    print(f"\nSample Prediction:")
    print(f"True class: {class_names[true_label]} ({true_label})")
    print(f"Predicted class: {class_names[predicted_class]} ({predicted_class})")
    print(f"Confidence: {confidence:.4f}")
    
    # Display the sample image
    plt.figure(figsize=(3, 3))
    plt.imshow(sample_image)
    plt.title(f"True: {class_names[true_label]}\nPredicted: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()
