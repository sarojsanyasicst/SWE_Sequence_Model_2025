# Practical-4 | DAM202

## Building a LSTM/GRU RNN for Time Series Weather Prediction: Guide

Based on weather dataset from Bangladesh (1990-2023), a comprehensive guide to build a RNN (LSTM) model for weather forecasting

## Framework Choice and Setup

For this task we are using **TensorFlow/Keras** as the primary framework because:

- Excellent RNN/LSTM support with easy-to-use APIs
- Built-in time series utilities and preprocessing tools
- Comprehensive documentation and community support for weather forecasting
- Easy deployment options for production systems

### Required Libraries Installation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```

## Data Preprocessing Pipeline

### 1. Data Loading and Initial Analysis

```python
# Load your Bangladesh weather data
def load_weather_data(file_path):
    """Load and prepare weather data for RNN processing"""
    df = pd.read_csv(file_path)

    # Assuming your data format: Year, Day, Wind_Speed, Specific_Humidity,
    # Relative_Humidity, Precipitation, Temperature
    columns = ['Year', 'Day', 'Wind_Speed', 'Specific_Humidity',
               'Relative_Humidity', 'Precipitation', 'Temperature']
    df.columns = columns

    # Create proper datetime index
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Day'].astype(str),
                                format='%Y-%j')
    df.set_index('Date', inplace=True)

    # Drop redundant columns
    df.drop(['Year', 'Day'], axis=1, inplace=True)

    return df

# Data exploration
def explore_data(df):
    """Comprehensive data exploration"""
    print("Dataset Shape:", df.shape)
    print("\nData Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Plot time series for all features
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(15, 3*len(df.columns)))
    for i, col in enumerate(df.columns):
        axes[i].plot(df.index, df[col])
        axes[i].set_title(f'{col} Over Time')
        axes[i].set_ylabel(col)
    plt.tight_layout()
    plt.show()
```

### 2. Advanced Data Cleaning and Feature Engineering

```python
def clean_and_engineer_features(df):
    """Advanced preprocessing for weather data"""

    # Handle missing values with forward fill and interpolation
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.interpolate(method='linear')

    # Remove outliers using IQR method
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Feature engineering for time series
    df['Day_of_Year'] = df.index.dayofyear
    df['Month'] = df.index.month
    df['Season'] = df['Month'].apply(get_season)

    # Cyclical encoding for temporal features[^4]
    df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Year'] / 365.25)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Year'] / 365.25)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Moving averages for trend capture
    for col in ['Temperature', 'Wind_Speed', 'Relative_Humidity']:
        df[f'{col}_MA_7'] = df[col].rolling(window=7).mean()
        df[f'{col}_MA_30'] = df[col].rolling(window=30).mean()

    # Lag features for temporal dependencies
    for lag in [1, 2, 3, 7, 30]:
        df[f'Temp_lag_{lag}'] = df['Temperature'].shift(lag)

    # Drop rows with NaN values after feature engineering
    df.dropna(inplace=True)

    return df

def get_season(month):
    """Convert month to season"""
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn
```

### 3. Data Normalization and Scaling

```python
def normalize_data(df, target_column='Temperature', scaler_type='minmax'):
    """
    Normalize data for RNN training
    Critical: Fit scaler only on training data to prevent data leakage[^21][^28]
    """

    # Separate features and target
    feature_columns = [col for col in df.columns if col != target_column]

    if scaler_type == 'minmax':
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    else:
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

    return df, feature_columns, feature_scaler, target_scaler

def create_train_test_split(df, test_size=0.2, val_size=0.1):
    """
    Time series aware train-test split
    No shuffling to maintain temporal order[^28]
    """
    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))

    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]

    return train_data, val_data, test_data
```

## RNN Model Architecture

### 1. Sequence Creation for Time Series

```python
def create_sequences(data, target, sequence_length, prediction_horizon=1):
    """
    Create sequences for RNN training

    Parameters:
    - sequence_length: Number of time steps to look back
    - prediction_horizon: Number of future steps to predict
    """
    X, y = [], []

    for i in range(sequence_length, len(data) - prediction_horizon + 1):
        X.append(data[i-sequence_length:i])
        y.append(target[i:i+prediction_horizon])

    return np.array(X), np.array(y)

def prepare_sequences(train_data, val_data, test_data, feature_columns,
                     target_column, sequence_length, prediction_horizon,
                     feature_scaler, target_scaler):
    """Prepare sequences with proper scaling"""

    # Fit scalers on training data only
    train_features_scaled = feature_scaler.fit_transform(train_data[feature_columns])
    train_target_scaled = target_scaler.fit_transform(train_data[[target_column]])

    # Transform validation and test data
    val_features_scaled = feature_scaler.transform(val_data[feature_columns])
    val_target_scaled = target_scaler.transform(val_data[[target_column]])

    test_features_scaled = feature_scaler.transform(test_data[feature_columns])
    test_target_scaled = target_scaler.transform(test_data[[target_column]])

    # Create sequences
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled.flatten(),
                                       sequence_length, prediction_horizon)
    X_val, y_val = create_sequences(val_features_scaled, val_target_scaled.flatten(),
                                   sequence_length, prediction_horizon)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled.flatten(),
                                     sequence_length, prediction_horizon)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

### 2. RNN Model Architecture

```python
def build_advanced_rnn_model(input_shape, model_type='LSTM',
                            hidden_units=[64, 32], dropout_rate=0.2,
                            learning_rate=0.001, prediction_horizon=1):
    """
    Build advanced RNN model with proper regularization

    Parameters:
    - model_type: 'SimpleRNN', 'LSTM', or 'GRU'
    - hidden_units: List of hidden layer sizes
    - dropout_rate: Dropout rate for regularization
    """

    model = Sequential(name=f"Weather_Prediction_{model_type}")

    # Input layer
    model.add(tf.keras.layers.Input(shape=input_shape, name='Input_Layer'))

    # Recurrent layers with dropout[^35]
    for i, units in enumerate(hidden_units):
        return_sequences = i < len(hidden_units) - 1

        if model_type == 'LSTM':
            model.add(LSTM(units, return_sequences=return_sequences,
                          dropout=dropout_rate, recurrent_dropout=dropout_rate,
                          name=f'LSTM_Layer_{i+1}'))
        elif model_type == 'GRU':
            model.add(GRU(units, return_sequences=return_sequences,
                         dropout=dropout_rate, recurrent_dropout=dropout_rate,
                         name=f'GRU_Layer_{i+1}'))
        else:
            model.add(SimpleRNN(units, return_sequences=return_sequences,
                               dropout=dropout_rate, recurrent_dropout=dropout_rate,
                               name=f'RNN_Layer_{i+1}'))

        # Additional dropout layer
        if return_sequences:
            model.add(Dropout(dropout_rate, name=f'Dropout_{i+1}'))

    # Dense layers for final prediction
    model.add(Dense(32, activation='relu', name='Dense_1'))
    model.add(Dropout(dropout_rate, name='Final_Dropout'))
    model.add(Dense(prediction_horizon, activation='linear', name='Output_Layer'))

    # Compile model with appropriate optimizer[^35]
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='mse',
                 metrics=['mae', 'mape'])

    return model

# Model comparison function
def compare_rnn_models(X_train, y_train, X_val, y_val, input_shape):
    """Compare different RNN architectures"""

    models = {}
    histories = {}

    for model_type in ['SimpleRNN', 'LSTM', 'GRU']:
        print(f"\nTraining {model_type} model...")

        model = build_advanced_rnn_model(input_shape, model_type=model_type)

        # Callbacks for better training[^35]
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss'),
            ModelCheckpoint(f'best_{model_type}_model.h5', save_best_only=True, monitor='val_loss')
        ]

        # Train model
        history = model.fit(X_train, y_train,
                           validation_data=(X_val, y_val),
                           epochs=100,
                           batch_size=32,
                           callbacks=callbacks,
                           verbose=1)

        models[model_type] = model
        histories[model_type] = history

    return models, histories
```

## Model Training and Hyperparameter Tuning

### 1. Advanced Training Setup

```python
def train_optimized_model(X_train, y_train, X_val, y_val, input_shape,
                         hyperparameters=None):
    """
    Train model with optimized hyperparameters[^35][^47]
    """

    if hyperparameters is None:
        # Default optimized hyperparameters based on research[^35]
        hyperparameters = {
            'model_type': 'LSTM',
            'hidden_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 150
        }

    model = build_advanced_rnn_model(
        input_shape,
        model_type=hyperparameters['model_type'],
        hidden_units=hyperparameters['hidden_units'],
        dropout_rate=hyperparameters['dropout_rate'],
        learning_rate=hyperparameters['learning_rate']
    )

    # Advanced callbacks for professional training
    callbacks = [
        EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor='val_loss',
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            factor=0.2,
            patience=8,
            min_lr=1e-7,
            monitor='val_loss',
            verbose=1
        ),
        ModelCheckpoint(
            'best_weather_model.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]

    # Train with validation monitoring
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=hyperparameters['epochs'],
        batch_size=hyperparameters['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    return model, history

def hyperparameter_optimization(X_train, y_train, X_val, y_val, input_shape):
    """
    Grid search for hyperparameter optimization[^35][^47]
    """

    param_grid = {
        'model_type': ['LSTM', 'GRU'],
        'hidden_units': [[64, 32], [128, 64], [128, 64, 32]],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.002],
        'batch_size': [32, 64, 128]
    }

    best_score = float('inf')
    best_params = None
    results = []

    from itertools import product

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    print(f"Testing {len(combinations)} hyperparameter combinations...")

    for i, params in enumerate(combinations[:10]):  # Limit for demo
        print(f"\nTesting combination {i+1}: {params}")

        try:
            model, history = train_optimized_model(X_train, y_train, X_val, y_val,
                                                 input_shape, params)

            # Get best validation loss
            val_loss = min(history.history['val_loss'])

            results.append({
                'params': params,
                'val_loss': val_loss,
                'model': model
            })

            if val_loss < best_score:
                best_score = val_loss
                best_params = params

        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

    return best_params, results
```

## Model Evaluation and Validation

### 1. Comprehensive Evaluation Metrics

```python
def evaluate_model_comprehensive(model, X_test, y_test, target_scaler,
                               test_dates=None):
    """
    Comprehensive model evaluation with multiple metrics[^22][^25]
    """

    # Make predictions
    y_pred_scaled = model.predict(X_test)

    # Inverse transform to original scale
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

    # Temperature-specific metrics
    temp_accuracy_1deg = np.mean(np.abs(y_test_original - y_pred_original) <= 1.0) * 100
    temp_accuracy_2deg = np.mean(np.abs(y_test_original - y_pred_original) <= 2.0) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Accuracy_1deg': temp_accuracy_1deg,
        'Accuracy_2deg': temp_accuracy_2deg
    }

    print("Model Evaluation Results:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Visualization
    plot_predictions(y_test_original, y_pred_original, test_dates)
    plot_residuals(y_test_original, y_pred_original)

    return metrics, y_pred_original

def plot_predictions(y_true, y_pred, dates=None):
    """Plot actual vs predicted values"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Time series plot
    if dates is not None:
        ax1.plot(dates, y_true, label='Actual', alpha=0.7)
        ax1.plot(dates, y_pred, label='Predicted', alpha=0.7)
    else:
        ax1.plot(y_true, label='Actual', alpha=0.7)
        ax1.plot(y_pred, label='Predicted', alpha=0.7)

    ax1.set_title('Temperature Prediction Results')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True)

    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Temperature (°C)')
    ax2.set_ylabel('Predicted Temperature (°C)')
    ax2.set_title('Actual vs Predicted Temperature')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred):
    """Plot residual analysis"""

    residuals = y_true - y_pred

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Temperature')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True)

    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True)

    # QQ plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    ax3.grid(True)

    # Time series of residuals
    ax4.plot(residuals)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals Over Time')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()
```

## Complete Implementation Pipeline

### 1. Main Training Pipeline

```python
def main_weather_prediction_pipeline(data_path, target_column='Temperature'):
    """
    Complete pipeline for weather prediction using RNN
    """

    print("Starting Weather Prediction Pipeline...")
    print("=" * 50)

    # 1. Load and explore data
    print("1. Loading and exploring data...")
    df = load_weather_data(data_path)
    explore_data(df)

    # 2. Clean and engineer features
    print("\n2. Cleaning and engineering features...")
    df_processed = clean_and_engineer_features(df)

    # 3. Prepare data splits
    print("\n3. Preparing data splits...")
    train_data, val_data, test_data = create_train_test_split(df_processed)

    # 4. Normalize data
    print("\n4. Normalizing data...")
    df_processed, feature_columns, feature_scaler, target_scaler = normalize_data(
        df_processed, target_column)

    # 5. Create sequences
    print("\n5. Creating sequences...")
    sequence_length = 30  # Use 30 days of history
    prediction_horizon = 1  # Predict 1 day ahead

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_sequences(
        train_data, val_data, test_data, feature_columns, target_column,
        sequence_length, prediction_horizon, feature_scaler, target_scaler
    )

    print(f"Training sequences shape: {X_train.shape}")
    print(f"Training targets shape: {y_train.shape}")

    # 6. Hyperparameter optimization (optional)
    print("\n6. Optimizing hyperparameters...")
    input_shape = (X_train.shape[^1], X_train.shape[^2])
    best_params, results = hyperparameter_optimization(X_train, y_train, X_val, y_val, input_shape)

    print(f"Best hyperparameters: {best_params}")

    # 7. Train final model
    print("\n7. Training final model...")
    final_model, history = train_optimized_model(X_train, y_train, X_val, y_val,
                                                input_shape, best_params)

    # 8. Evaluate model
    print("\n8. Evaluating model...")
    test_dates = test_data.index[sequence_length:]
    metrics, predictions = evaluate_model_comprehensive(final_model, X_test, y_test,
                                                      target_scaler, test_dates)

    # 9. Plot training history
    plot_training_history(history)

    # 10. Save model and scalers
    print("\n9. Saving model and preprocessors...")
    final_model.save('weather_prediction_model.h5')

    import joblib
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')

    print("\nPipeline completed successfully!")

    return final_model, metrics, feature_scaler, target_scaler

def plot_training_history(history):
    """Plot training history"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
```

### 2. Prediction Functions

```python
def make_future_predictions(model, last_sequence, feature_scaler, target_scaler,
                           days_ahead=7):
    """
    Make future predictions for specified days
    """

    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days_ahead):
        # Predict next day
        pred_scaled = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        predictions.append(pred_original)

        # Update sequence for next prediction
        # This is a simplified approach - in practice, you'd need to update all features
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, -1] = pred_scaled[0, 0]  # Update temperature feature

    return predictions

def predict_weather(model_path, scaler_paths, new_data, sequence_length=30):
    """
    Load saved model and make predictions on new data
    """

    import joblib

    # Load model and scalers
    model = tf.keras.models.load_model(model_path)
    feature_scaler = joblib.load(scaler_paths['feature_scaler'])
    target_scaler = joblib.load(scaler_paths['target_scaler'])

    # Preprocess new data
    new_data_scaled = feature_scaler.transform(new_data)

    # Create sequence
    if len(new_data_scaled) >= sequence_length:
        sequence = new_data_scaled[-sequence_length:]

        # Make prediction
        prediction_scaled = model.predict(sequence.reshape(1, sequence_length, -1))
        prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]

        return prediction
    else:
        raise ValueError(f"Need at least {sequence_length} days of data for prediction")
```

## Usage Example

```python
# Run the complete pipeline
if __name__ == "__main__":
    # Path to your Bangladesh weather data CSV file
    data_path = "bangladesh_weather_data.csv"

    # Run the complete pipeline
    model, metrics, feature_scaler, target_scaler = main_weather_prediction_pipeline(data_path)

    # Make future predictions
    # You would need to provide the last sequence from your test data
    # future_predictions = make_future_predictions(model, last_sequence,
    #                                            feature_scaler, target_scaler, days_ahead=7)

    print("Weather prediction model training completed!")
```

## Best Practices and Professional Tips

### 1. Model Performance Optimization

- **Hyperparameter Tuning**: Use techniques like Bayesian optimization for better hyperparameter search
- **Regularization**: Implement dropout, early stopping, and learning rate scheduling to prevent overfitting
- **Architecture Selection**: Compare SimpleRNN, LSTM, and GRU to find the best architecture for your data

### 2. Data Quality and Preprocessing

- **Normalization**: Always fit scalers on training data only to prevent data leakage
- **Feature Engineering**: Create temporal features, moving averages, and lag variables
- **Missing Data**: Handle missing values appropriately with interpolation or forward filling

### 3. Evaluation and Validation

- **Multiple Metrics**: Use RMSE, MAE, MAPE, and domain-specific accuracy measures
- **Residual Analysis**: Analyze prediction errors to identify model weaknesses
- **Cross-validation**: Use time series cross-validation for robust performance estimates

### 4. Production Deployment

- **Model Monitoring**: Implement continuous monitoring of model performance
- **Data Pipeline**: Create automated data preprocessing pipelines
- **API Integration**: Build REST APIs for real-time predictions
- **Retraining**: Set up automated model retraining with new data

This comprehensive guide provides you with a professional-grade RNN implementation for weather prediction. The code includes all necessary components for data preprocessing, model training, evaluation, and deployment. You can customize the hyperparameters and architecture based on your specific requirements and computational resources.
