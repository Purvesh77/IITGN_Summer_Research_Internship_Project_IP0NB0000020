# Tap Detection Using Simulated Accelerometer Data

This document provides a detailed explanation of the process used to detect taps using simulated accelerometer data. The process involves data simulation, visualization, machine learning model training, and real-time tap detection.

## Steps Overview

1. **Data Simulation**
2. **Data Visualization**
3. **Machine Learning Model Training**
4. **Real-Time Tap Detection**

## 1. Data Simulation

The first step is to simulate accelerometer data. This is done using a function that generates random noise to mimic the behavior of an accelerometer. Additionally, spikes are added to the data to simulate taps.

### Code Explanation

```python
def simulate_data(duration=10, frequency=100):
    time_data = np.linspace(0, duration, duration * frequency)
    x_data = np.random.normal(0, 1, len(time_data))
    y_data = np.random.normal(0, 1, len(time_data))
    z_data = np.random.normal(0, 1, len(time_data))
    
    # Simulate taps by adding spikes
    for _ in range(5):  # Simulate 5 taps
        tap_time = np.random.randint(0, len(time_data))
        x_data[tap_time:tap_time+5] += np.random.normal(10, 2, 5)
        y_data[tap_time:tap_time+5] += np.random.normal(10, 2, 5)
        z_data[tap_time:tap_time+5] += np.random.normal(10, 2, 5)
    
    return time_data, x_data, y_data, z_data
```
- Duration and Frequency: The data is generated for a specified duration and frequency, simulating a real-time data stream.
- Random Noise: Normal distribution is used to generate random noise for X, Y, and Z axes.
- Spikes for Taps: Random spikes are added to simulate taps, which are crucial for training the ML model.

## 2. Data Visualization

Visualizing the data helps in understanding the patterns and identifying spikes that correspond to taps.

### Code Explanation
```python
def plot_data(time_data, x_data, y_data, z_data):
    plt.figure(figsize=(12, 6))
    plt.plot(time_data, x_data, label='X')
    plt.plot(time_data, y_data, label='Y')
    plt.plot(time_data, z_data, label='Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.title('Accelerometer Data from CSV')
    plt.legend()
    plt.show()
```
- Plotting: The function uses matplotlib to plot the X, Y, and Z data over time, allowing visual identification of spikes.

## 3. Machine Learning Model Training

The core of the tap detection system is the machine learning model. We use a Random Forest classifier to distinguish between tap and no-tap states.

Why Random Forest?
- Robustness: Random Forest is an ensemble method that combines multiple decision trees, making it robust to overfitting and noise.
- Feature Importance: It provides insights into feature importance, which can be useful for understanding which axis contributes most to tap detection.
- Non-linearity: It can capture non-linear relationships in the data, which is beneficial given the complexity of accelerometer data.

### Code Explanation
```python
def train_model(x_data, y_data, z_data):
    X = np.column_stack((x_data, y_data, z_data))
    y = np.array([1 if max(row) > 8 else 0 for row in X])  # Label spikes as taps
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model accuracy:", model.score(X_test, y_test))
    return model
```
- Feature Matrix (X): The X, Y, and Z data are combined into a feature matrix.
- Labeling (y): A simple threshold is used to label data points as taps (1) or no-taps (0).
- Model Training: The data is split into training and testing sets. The Random Forest model is trained on the training set and evaluated on the test set.

## 4. Real-Time Tap Detection
The trained model is used to simulate real-time tap detection by predicting each data point.

### Code Explanation
```Python
def detect_tap(model, x_data, y_data, z_data):
    X = np.column_stack((x_data, y_data, z_data))
    for i in range(len(X)):
        prediction = model.predict([X[i]])
        if prediction == 1:
            print(f"Tap detected at index {i}!")
```
- Prediction: The model predicts each data point, and if a tap is detected, it prints "Tap detected!" along with the index.

## Conclusion
This process demonstrates how to simulate accelerometer data, visualize it, train a machine learning model, and perform real-time tap detection. The use of a Random Forest classifier provides a robust and effective method for distinguishing between tap and no-tap states.
