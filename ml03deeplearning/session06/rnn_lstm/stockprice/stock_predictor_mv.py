'''
the Code to Include More Features
We’ll change the model to a multivariate time series model, where multiple input features
(like Open, High, Low, Volume) are used to predict Adj Close.

Changes Required:
- Modify load_and_preprocess_data to include new columns.
- Update _create_sequences to generate time-windowed sequences for all features.
- Adjust reshaping and model input shape accordingly.

How the Updated Multivariate Model Works
- Input Shape: Each sample is now a 2D matrix of shape (window_size, num_features) — e.g., 45 days × 4 features.
- Model: The LSTM layers learn dependencies not just across time, but also between features like Open, High, Low, and Volume.
- Advantage: It provides more context to the model, which can result in better accuracy when trends in price
  correlate with volume or price range volatility.

Summary of Major Enhancements
Feature	Original	Updated
Input Data	Adj Close only	Open, High, Low, Volume
Model Type	Univariate LSTM	Multivariate LSTM
Input Shape	(samples, window, 1)	(samples, window, 4)
Potential Accuracy	Limited by single feature	Enhanced with more market context

=====================================================

The multivariate version of the model provides several key advantages over the univariate model,
especially when dealing with complex time series data like stock prices:

1. More Contextual Information
- Univariate models only learn patterns from a single variable (e.g., past adjusted close prices).
- Multivariate models use multiple inputs (e.g., Open, High, Low, Volume, RSI, MACD), giving the model a
  richer context about what’s happening in the market.

Advantage: This allows the model to understand price dynamics, market volatility, and trading volume
pressure, which can improve forecasting accuracy.

2. Improved Predictive Power
- Multivariate models can uncover relationships between features and the target variable
  (e.g., how volume spikes precede price moves).
- The model can learn from leading indicators or lagging signals.

Advantage: Better generalization and often lower prediction error, especially in noisy or nonlinear
data like stock markets.

3. Capturing Feature Interactions
- Time series often have interdependencies: a low closing price with high volume might mean something
  different than a low close with low volume.
- Multivariate LSTMs can model these feature interactions over time.

Advantage: Helps the model capture more nuanced patterns, rather than just memorizing sequences of one variable.

4. Flexibility to Add Technical Indicators
- You can easily integrate domain-specific knowledge (e.g., RSI, MACD, Bollinger Bands) as features.
- These indicators are derived from price/volume trends and help the model interpret market signals.

Advantage: Gives your model access to engineered features that are proven predictors in traditional finance.

5. Better Performance in Regime Changes
- Stock markets often behave differently under varying regimes (bull vs bear markets, low vs high volatility).
- Univariate models may overfit to recent trends.
- Multivariate models, by having broader inputs, can adapt more easily to changing market conditions.
Advantage: More robust to market shocks, news events, or structural breaks in the data.

When Univariate Might Still Be Preferred:
- Data is limited or only one meaningful feature is available.
- Goal is interpretability or simplicity.
- You’re doing very short-term forecasting or anomaly detection.

In Summary:
Aspect	Univariate	Multivariate
Features Used	Only one	Multiple (OHLCV, RSI, MACD, etc.)
Context Awareness	Limited	Richer contextual understanding
Predictive Accuracy	Usually lower	Often significantly higher
Risk of Overfitting	Lower	Higher, if not handled with care
Suitable for Complex Data?	Not ideal	Yes, especially with financial time series

'''

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint


class StockPricePredictor:
    def __init__(self, data_path, window_size=45, test_split_date='2022-08-14',
                 random_seed=12345, model_filename='GoogModel.keras'):
        self.data_path = data_path
        self.window_size = window_size
        self.test_split_date = test_split_date
        self.random_seed = random_seed
        self.model_filename = model_filename
        self.df = None
        self.df_scaled = None
        self.scaler = None
        self.model = None
        self.feature_cols = ['Open', 'High', 'Low', 'Volume']
        self.target_col = 'Adj Close'
        self.X_all = None
        self.y_all = None
        self.dates_all = None

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.data_path, parse_dates=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format="%Y-%m-%d")

        # Drop missing values just in case
        self.df.dropna(subset=self.feature_cols + [self.target_col], inplace=True)

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.df[self.feature_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=self.feature_cols)
        scaled_df['y'] = self.df[self.target_col].values
        scaled_df['Date'] = self.df['Date'].values

        self.df_scaled = scaled_df.copy()
        self._create_multivariate_sequences()

    def _create_multivariate_sequences(self):
        sequences = []
        targets = []
        dates = []

        data = self.df_scaled[self.feature_cols].values
        target = self.df_scaled['y'].values
        all_dates = self.df_scaled['Date'].values

        for i in range(self.window_size, len(data)):
            sequences.append(data[i - self.window_size:i])
            targets.append(target[i])
            dates.append(all_dates[i])

        self.X_all = np.array(sequences)
        self.y_all = np.array(targets)
        self.dates_all = np.array(dates)

    def split_data(self):
        split_idx = np.where(self.dates_all <= np.datetime64(self.test_split_date))[0][-1]

        X_train = self.X_all[:split_idx + 1]
        y_train = self.y_all[:split_idx + 1]
        X_test = self.X_all[split_idx + 1:]
        y_test = self.y_all[split_idx + 1:]

        return X_train, y_train, X_test, y_test

    def build_lstm_model(self):
        input_shape = (self.window_size, len(self.feature_cols))
        self.model = Sequential()
        self.model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(128, activation='relu'))
        self.model.add(Dense(units=1, activation='linear'))
        self.model.compile(optimizer='adam', loss='huber', metrics=['mse'])
        self.model.summary()

    def train_model(self, X_train, y_train, X_test, y_test, epochs=200, batch_size=32):
        checkpoint = ModelCheckpoint(
            self.model_filename,
            monitor='val_mse',
            save_best_only=True,
            verbose=1
        )
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint]
        )
        self.model = tf.keras.models.load_model(self.model_filename)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_model(self, y_true, y_pred, set_name=""):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        accuracy = np.round(100 - mape * 100, 2)
        print(f"{set_name} set accuracy: {accuracy}%")
        return mape

    def plot_predictions(self, dates, actuals, predictions, title="Actual vs Predicted", y_label="Stock Price"):
        plt.figure(figsize=(20, 6))
        plt.plot(dates, actuals, label='Actual', color='blue')
        plt.plot(dates, predictions, label='Predicted', color='red')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    def run_prediction_pipeline(self, epochs=200):
        self.load_and_preprocess_data()
        X_train, y_train, X_test, y_test = self.split_data()
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
        self.build_lstm_model()
        self.train_model(X_train, y_train, X_test, y_test, epochs=epochs)

        y_tr_pred = self.predict(X_train)
        y_ts_pred = self.predict(X_test)

        self.evaluate_model(y_train, y_tr_pred, "Training")
        self.evaluate_model(y_test, y_ts_pred, "Test")

        self.plot_predictions(self.dates_all[:len(y_tr_pred)], y_train, y_tr_pred, "Training Predictions")
        self.plot_predictions(self.dates_all[len(y_tr_pred):], y_test, y_ts_pred, "Test Predictions")


if __name__ == "__main__":
    predictor = StockPricePredictor('GOOG3.csv')
    predictor.run_prediction_pipeline(epochs=200)
