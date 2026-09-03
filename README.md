# Machine Temperature Prediction using LSTM

## About the Project

This project predicts the next temperature of a machine using an LSTM (Long Short-Term Memory) neural network.

The model learns from previous temperature readings and uses them to predict the upcoming temperature.

## Problem Statement

Machine temperature can change over time. If the temperature becomes too high, it may indicate that the machine is under excessive load or may develop a fault.

The goal of this project is to predict the next temperature reading so that abnormal temperature changes can be detected earlier.

## How It Works

The project follows these steps:

1. Generate machine temperature readings.
2. Prepare the temperature data.
3. Create sequences using the previous 5 readings.
4. Train an LSTM model.
5. Evaluate the model.
6. Predict the next machine temperature.

### Example

The model receives the previous 5 readings:

```text
39.95, 39.50, 39.96, 39.97, 39.54
