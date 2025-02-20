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
