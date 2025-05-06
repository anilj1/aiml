'''
Explanation of Changes and Improvements:

Class Structure: The code is now organized within a Python class StockPricePredictor. This encapsulates the data,
methods, and logic related to stock price prediction, making the code more modular and reusable.

Initialization (__init__): The constructor initializes key parameters as instance variables, such as the data path,
window size, test split date, random seed, and model filename.

Data Loading and Preprocessing (load_and_preprocess_data): This method handles loading the CSV file, converting the
'Date' column to datetime objects, selecting the 'Date' and 'Adj Close' columns, renaming 'Adj Close' to 'y',
sorting by date, and calling the sequence creation method. It also handles dropping NaN values.

Sequence Creation (_create_sequences): This private method (indicated by the leading underscore) creates the lagged
features required for the LSTM model based on the window_size.

Data Splitting (split_data): This method splits the data into training and testing sets based on the test_split_date.
It also performs feature scaling using StandardScaler and reshapes the data to the 3D format expected by LSTM layers
([samples, time_steps, features]).

Model Building (build_lstm_model): This method defines and compiles the LSTM model. The model architecture
(two LSTM layers and a Dense output layer) and the optimizer, loss function, and metrics are defined here.

Model Training (train_model): This method trains the LSTM model using the provided training data. It incorporates
a ModelCheckpoint callback to save the best model based on the validation mean squared error. It also loads the
best saved model after training.

Prediction (predict): This method takes input features and uses the trained model to make predictions.

Model Evaluation (evaluate_model): This method calculates and prints the Mean Absolute Percentage Error (MAPE)
as a measure of the model's accuracy on the given data.

Plotting Predictions (plot_predictions): This method generates a line plot comparing the actual and predicted
stock prices over time.

Running the Pipeline (run_prediction_pipeline): This method orchestrates the entire process, calling the
data loading, preprocessing, splitting, model building, training, prediction, and evaluation methods in sequence.

if __name__ == "__main__": Block: This ensures that the code to create and run the StockPricePredictor
instance is only executed when the script is run directly (not when imported as a module).

Benefits of Refactoring:

Organization: The code is now logically structured within a class, making it easier to understand and maintain.
Reusability: The StockPricePredictor class can be easily instantiated and used for different stock datasets
or with different configurations.
Modularity: Each step of the prediction process is encapsulated in a separate method, improving code readability
and making it easier to modify individual components.
Clarity: The code is more readable with descriptive method names and clear separation of concerns.
Parameterization: Key parameters like window_size, test_split_date, and epochs are now configurable through the
class constructor or method arguments.
This refactored code provides a more robust and maintainable structure for your stock price prediction project.
'''

'''
Why the Original Code Uses Only Adjusted Close
The original code uses only the “Adjusted Close” value because it follows a univariate (single variate) 
time series approach. Here's what that means:

Univariate model: A model that uses only one variable (feature) to predict future values of that same variable.

Adjusted Close is often preferred in finance because it reflects stock splits, dividends, and other corporate 
actions that affect the stock price. It's more accurate for historical price modeling.

This approach simplifies the problem and is suitable for many use cases in time-series forecasting, especially when:
- You want to model temporal patterns purely from past price behavior.
- You're avoiding the complexity of multivariate relationships.

How the Model Works
- Windowing: The model uses a sliding window (e.g., last 45 days) of adjusted close values to predict the next value.
- Sequence Creation: These windows are turned into feature vectors (e.g., y1, y2, ..., y45) where the target is y.
- LSTM Model: Long Short-Term Memory networks are well-suited for sequential data and learn temporal dependencies in the stock prices.
- Normalization: Features are scaled with StandardScaler to stabilize training.
'''

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_percentage_error


class StockPricePredictor:
    """
    A class for predicting stock prices using an LSTM model.
    """

    def __init__(self, data_path, window_size=45, test_split_date='2022-08-14', random_seed=12345,
                 model_filename='GoogModel.keras'):
        """
        Initializes the StockPricePredictor.

        Args:
            data_path (str): Path to the CSV file containing the stock data.
            window_size (int): The number of previous days to consider for prediction.
            test_split_date (str): The date to split the data into training and testing sets (YYYY-MM-DD).
            random_seed (int): Seed for random number generation to ensure reproducibility.
            model_filename (str): Filename for saving the best model.
        """
        self.data_path = data_path
        self.window_size = window_size
        self.test_split_date = test_split_date
        self.random_seed = random_seed
        self.model_filename = model_filename
        self.df = None
        self.df_close = None
        self.scaler = None
        self.model = None
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def load_and_preprocess_data(self):
        """
        Loads the data, preprocesses it, and creates sequences for LSTM.
        """
        self.df = pd.read_csv(self.data_path, parse_dates=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format="%Y-%m-%d")
        self.df_close = self.df[['Date', 'Adj Close']].copy()
        self.df_close.columns = ['Date', 'y']
        self.df_close.sort_values(by='Date', inplace=True)
        self._create_sequences()
        self.df_close.dropna(axis=0, inplace=True)

    def _create_sequences(self):
        """
        Creates lagged features based on the specified window size.
        """
        for i in range(self.window_size):
            self.df_close[f'y{i + 1}'] = self.df_close['y'].shift(i + 1)

    def split_data(self):
        """
        Splits the data into training and testing sets.

        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        df_train = self.df_close[self.df_close['Date'] <= pd.to_datetime(self.test_split_date)].copy()
        df_test = self.df_close[self.df_close['Date'] > pd.to_datetime(self.test_split_date)].copy()

        X_train = df_train.drop(['Date', 'y'], axis=1)
        y_train = df_train['y']
        X_test = df_test.drop(['Date', 'y'], axis=1)
        y_test = df_test['y']

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        return X_train, y_train, X_test, y_test

    def build_lstm_model(self):
        """
        Builds the LSTM model.
        """
        self.model = Sequential()
        self.model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(None, 1)))
        self.model.add(LSTM(128, activation='relu'))
        self.model.add(Dense(units=1, activation='linear'))
        self.model.compile(optimizer='adam', loss='huber', metrics=['mse'])
        self.model.summary()

    def train_model(self, X_train, y_train, X_test, y_test, epochs=200, batch_size=32):
        """
        Trains the LSTM model.

        Args:
            X_train (np.array): Training features.
            y_train (pd.Series): Training target.
            X_test (np.array): Testing features.
            y_test (pd.Series): Testing target.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
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
        """
        Makes predictions using the loaded model.

        Args:
            X (np.array): Input features for prediction.

        Returns:
            np.array: Predicted values.
        """
        return self.model.predict(X)

    def evaluate_model(self, y_true, y_pred, set_name=""):
        """
        Evaluates the model using Mean Absolute Percentage Error (MAPE).

        Args:
            y_true (pd.Series or np.array): True values.
            y_pred (np.array): Predicted values.
            set_name (str): Name of the dataset (e.g., "Training", "Test").

        Returns:
            float: MAPE score.
        """
        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        accuracy = np.round(100 - mape * 100, 2)
        print(f"{set_name} set accuracy: {accuracy}%")
        return mape

    def plot_predictions(self, df_actual, y_pred, title="Stock Price - Actual vs Predicted", y_label="Stock Price"):
        """
        Plots the actual vs predicted values.

        Args:
            df_actual (pd.DataFrame): DataFrame containing 'Date' and actual 'y' values.
            y_pred (np.array): Predicted values.
            title (str): Title of the plot.
            y_label (str): Label for the y-axis.
        """
        plt.figure(figsize=(20, 6))
        plt.plot(df_actual['Date'], df_actual['y'], color='blue', label='Actual')
        plt.plot(df_actual['Date'], y_pred, color='red', label='Predicted')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    def run_prediction_pipeline(self, epochs=200):
        """
        Runs the complete stock price prediction pipeline.

        Args:
            epochs (int): Number of training epochs.
        """
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

        df_train_results = self.df_close[self.df_close['Date'] <= pd.to_datetime(self.test_split_date)][
            ['Date', 'y']].copy()
        df_train_results['Pred'] = y_tr_pred
        self.plot_predictions(df_train_results, y_tr_pred,
                              title="Google Stock Price - Actual vs Predicted (Training Set)")

        df_test_results = self.df_close[self.df_close['Date'] > pd.to_datetime(self.test_split_date)][
            ['Date', 'y']].copy()
        df_test_results['yhat'] = y_ts_pred
        self.plot_predictions(df_test_results, y_ts_pred, title="Google Stock Price - Actual vs Predicted (Test Set)")


if __name__ == "__main__":
    predictor = StockPricePredictor('GOOG3.csv')
    predictor.run_prediction_pipeline(epochs=200)
