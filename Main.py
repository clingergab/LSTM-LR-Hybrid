# I'll use LSTM to capture time dependent patterns and LR for trends
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


SEQ_LENGTH = 60

data = pd.read_csv('Data/apple_stock_data.csv')

# set Data to datetime and set as index
data['Data'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.index = pd.to_datetime(data.index)  # Ensure index is datetime

data = data[['Close']]

# scale data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

# prepare data for LSTM by creating sequences with defined length to predict next day
def create_sequences(data: np.array, length: int = SEQ_LENGTH) -> np.array: 
    # create arrays containing data for defined length and corresponding output
    X, y = [], []
    for i in range(len(data) - length):
        X.append(data[i: i + length])
        y.append(data[i + length])
        
    return np.array(X), np.array(y)

X, y = create_sequences(data['Close'].values)
# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

lstm_model = Sequential()
# lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# lstm_model.add(LSTM(units=50))
# lstm_model.add(Dense(1))
# lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # First LSTM layer
lstm_model.add(Dropout(0.2))  # Add dropout to prevent overfitting
lstm_model.add(LSTM(units=100))  # Second LSTM layer
lstm_model.add(Dropout(0.2))  # Add dropout again
lstm_model.add(Dense(1))  # Output layer

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)
# lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("LSTM Training Progress")
plt.show()


data['lag1'] = data['Close'].shift(1)
data['lag2'] = data['Close'].shift(2)
data['lag3'] = data['Close'].shift(3)
data = data.dropna()

X_lin = data[['lag1', 'lag2', 'lag3']]
y_lin = data['Close']
X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin, y_lin, test_size=0.2, train_size=0.8, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_lin_train, y_lin_train)

X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
lstm_pred = lstm_model.predict(X_test_lstm)
lstm_pred_trans = scaler.inverse_transform(lstm_pred)

# True values (ensure they're in the original scale if predictions are inverse-transformed)
y_true_lstm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae_lstm = mean_absolute_error(y_true_lstm, lstm_pred_trans.flatten())
mse_lstm = mean_squared_error(y_true_lstm, lstm_pred_trans.flatten())
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_true_lstm, lstm_pred_trans.flatten())
print("LSTM Metrics: MAE =", mae_lstm, ", MSE =", mse_lstm, ", RMSE =", rmse_lstm, ", R2 =", r2_lstm)


lin_pred = lin_model.predict(X_lin_test)
lin_pred_trans = scaler.inverse_transform(lin_pred.reshape(-1, 1))

y_true_lin = scaler.inverse_transform(y_lin_test.values.reshape(-1, 1)).flatten()

mae_lin = mean_absolute_error(y_true_lin, lin_pred_trans.flatten())
mse_lin = mean_squared_error(y_true_lin, lin_pred_trans.flatten())
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_true_lin, lin_pred_trans.flatten())
print("Linear Regression Metrics: MAE =", mae_lin, ", MSE =", mse_lin, ", RMSE =", rmse_lin, ", R2 =", r2_lin)

# Align both test sets to the smallest common length
common_length = min(len(X_test), len(X_lin_test))

# Adjust LSTM predictions
lstm_pred = lstm_pred[:common_length, :]
lstm_pred_final = lstm_pred[:, -1]  # Use the last time step

# Adjust Linear Regression predictions
lin_pred = lin_pred[:common_length].flatten()

# Combine predictions
hybrid_pred = (lstm_pred_final * 0.7) + (lin_pred * 0.3)
hybrid_pred_trans = scaler.inverse_transform(hybrid_pred.reshape(-1, 1))

# Hybrid
hybrid_true = y_true_lstm  
mae_hybrid = mean_absolute_error(hybrid_true, hybrid_pred_trans.flatten())
mse_hybrid = mean_squared_error(hybrid_true, hybrid_pred_trans.flatten())
rmse_hybrid = np.sqrt(mse_hybrid)
r2_hybrid = r2_score(hybrid_true, hybrid_pred_trans.flatten())
print("Hybrid Metrics: MAE =", mae_hybrid, ", MSE =", mse_hybrid, ", RMSE =", rmse_hybrid, ", R2 =", r2_hybrid)

plt.figure(figsize=(12, 6))
plt.plot(y_true_lstm, label='True Values', color='blue')
plt.plot(lstm_pred_trans.flatten(), label='LSTM Predictions', color='orange')
plt.plot(lin_pred_trans.flatten(), label='Linear Regression Predictions', color='green')
plt.plot(hybrid_pred_trans.flatten(), label='Hybrid Predictions', color='red')
plt.legend()
plt.title("True Values vs Predictions")
plt.show()

# predict the next 10 days
lstm_future_pred = []
last_seq = X[-1].reshape(1, SEQ_LENGTH, 1)
for _ in range(10):
    lstm_predict = lstm_model.predict(last_seq)[0, 0]
    lstm_future_pred.append(lstm_predict)
    lstm_pred_reshaped = lstm_predict.reshape(1, 1, 1)
    last_seq = np.append(last_seq[:, 1:, :], lstm_pred_reshaped, axis=1)
    
lstm_future_pred_trans = scaler.inverse_transform(np.array(lstm_future_pred).reshape(-1, 1))

lin_future_pred = []
lin_data = data['Close'].values[-3:]
for _ in range(10):
    lin_predict = lin_model.predict(lin_data.reshape(1, -1))[0]
    lin_future_pred.append(lin_predict)
    lin_data = np.append(lin_data[1:], lin_predict)
    
lin_future_pred_trans = scaler.inverse_transform(np.array(lin_future_pred).reshape(-1, 1))

future_hybrid_pred = (lstm_future_pred_trans * 0.7) + (lin_future_pred_trans * 0.3)

future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10)
pred_df = pd.DataFrame({
    'Date': future_dates,
    'LSTM Predictions': lstm_future_pred_trans.flatten(),
    'LR Predictions': lin_future_pred_trans.flatten(),
    'Hybrid Predictions': future_hybrid_pred.flatten()
})

print(pred_df)