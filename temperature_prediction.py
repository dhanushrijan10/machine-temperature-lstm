
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ============================================================
# STEP 1: CREATE MACHINE TEMPERATURE DATA
# ============================================================

np.random.seed(42)

temperature = np.array([
    30, 31, 32, 33, 34,
    35, 36, 37, 38, 39,
    40, 41, 42, 43, 44,
    45, 46, 47, 48, 49,
    50, 51, 52, 53, 54,
    55, 56, 57, 58, 59,
    60, 61, 62, 63, 64
])

# Create 200 temperature readings
base_temperature = np.linspace(30, 40, 200)

noise = np.random.normal(
    0,
    0.4,
    200
)

temperature = base_temperature + noise

print("Temperature readings:")
print(temperature)

print("\nNumber of readings:", len(temperature))


# ============================================================
# STEP 2: VISUALIZE TEMPERATURE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(temperature)

plt.title("Machine Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")

plt.grid(True)

plt.show()


# ============================================================
# STEP 3: TRAIN / TEST SPLIT
# ============================================================

train_size = int(len(temperature) * 0.80)

train_temperature = temperature[:train_size]

test_temperature = temperature[train_size:]

print("\n--------------------------------")
print("TRAIN / TEST SPLIT")
print("--------------------------------")

print("Total readings :", len(temperature))
print("Training data  :", len(train_temperature))
print("Testing data   :", len(test_temperature))


# ============================================================
# STEP 4: NORMALIZE DATA
# ============================================================

scaler = MinMaxScaler()

# IMPORTANT:
# Fit scaler ONLY on training data

train_scaled = scaler.fit_transform(
    train_temperature.reshape(-1, 1)
)

# Transform test data using the SAME scaler

test_scaled = scaler.transform(
    test_temperature.reshape(-1, 1)
)


# ============================================================
# STEP 5: CREATE SEQUENCES
# ============================================================

sequence_length = 5


def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(len(data) - sequence_length):

        X.append(
            data[i:i + sequence_length]
        )

        y.append(
            data[i + sequence_length]
        )

    return np.array(X), np.array(y)


# Training sequences

X_train, y_train = create_sequences(
    train_scaled,
    sequence_length
)


# ============================================================
# STEP 6: CREATE TEST SEQUENCES
# ============================================================

# We need the last 5 training readings as context
# for the beginning of the test set.

combined_test_data = np.concatenate(
    (
        train_scaled[-sequence_length:],
        test_scaled
    )
)

X_test, y_test = create_sequences(
    combined_test_data,
    sequence_length
)


print("\n--------------------------------")
print("DATA PREPARATION")
print("--------------------------------")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# STEP 7: BUILD LSTM MODEL
# ============================================================

model = Sequential()

model.add(
    LSTM(
        50,
        input_shape=(sequence_length, 1)
    )
)

model.add(
    Dense(1)
)


# ============================================================
# STEP 8: COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# ============================================================
# STEP 9: TRAIN MODEL
# ============================================================

print("\n--------------------------------")
print("TRAINING LSTM")
print("--------------------------------")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=1
)


# ============================================================
# STEP 10: PREDICT TEST DATA
# ============================================================

print("\n--------------------------------")
print("TESTING MODEL")
print("--------------------------------")

predictions_scaled = model.predict(
    X_test,
    verbose=0
)


# Convert predictions back to Celsius

predictions = scaler.inverse_transform(
    predictions_scaled
)

actual = scaler.inverse_transform(
    y_test
)


# ============================================================
# STEP 11: CALCULATE MAE
# ============================================================

mae = mean_absolute_error(
    actual,
    predictions
)


# ============================================================
# STEP 12: CALCULATE RMSE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)


print("\nModel Performance")

print("MAE :", mae, "°C")
print("RMSE:", rmse, "°C")


# ============================================================
# STEP 13: SHOW SOME PREDICTIONS
# ============================================================

print("\n--------------------------------")
print("SAMPLE PREDICTIONS")
print("--------------------------------")

for i in range(10):

    print(
        f"Example {i + 1}: "
        f"Actual = {actual[i][0]:.2f} °C, "
        f"Predicted = {predictions[i][0]:.2f} °C"
    )


# ============================================================
# STEP 14: PLOT ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    actual,
    label="Actual Temperature"
)

plt.plot(
    predictions,
    label="Predicted Temperature"
)

plt.title(
    "Actual vs Predicted Machine Temperature"
)

plt.xlabel("Time")

plt.ylabel("Temperature (°C)")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# STEP 15: PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"]
)

plt.title(
    "LSTM Training Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.show()


# ============================================================
# STEP 16: PREDICT NEXT TEMPERATURE
# ============================================================

last_sequence = temperature[-sequence_length:]

last_sequence_scaled = scaler.transform(
    last_sequence.reshape(-1, 1)
)

last_sequence_scaled = last_sequence_scaled.reshape(
    1,
    sequence_length,
    1
)


next_temperature_scaled = model.predict(
    last_sequence_scaled,
    verbose=0
)


next_temperature = scaler.inverse_transform(
    next_temperature_scaled
)


print("\n--------------------------------")
print("NEXT TEMPERATURE PREDICTION")
print("--------------------------------")

print(
    "Last 5 actual temperatures:"
)

print(last_sequence)

print(
    "\nPredicted next temperature:"
)

print(
    f"{next_temperature[0][0]:.2f} °C"
)


# ============================================================
# PROJECT COMPLETE
# ============================================================

print("\n================================")
print("MACHINE TEMPERATURE LSTM COMPLETE")
print("================================")