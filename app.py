import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Ensure model file exists before loading
MODEL_PATH = "stock_dl_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it is in the correct directory.")

# Load the updated model
model = load_model(MODEL_PATH, compile=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock', 'POWERGRID.NS')  # Default stock if none is entered

        try:
            # Define stock data range
            start = dt.datetime(2000, 1, 1)
            end = dt.datetime(2024, 10, 1)

            # Fetch stock data
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                return render_template("index.html", error="Invalid stock ticker or no data available.")

            # Descriptive statistics
            data_desc = df.describe()

            # Compute Exponential Moving Averages
            ema20 = df['Close'].ewm(span=20, adjust=False).mean()
            ema50 = df['Close'].ewm(span=50, adjust=False).mean()
            ema100 = df['Close'].ewm(span=100, adjust=False).mean()
            ema200 = df['Close'].ewm(span=200, adjust=False).mean()

            # Split Data
            data_training = df['Close'][0:int(len(df) * 0.70)]
            data_testing = df['Close'][int(len(df) * 0.70):]

            # Scaling data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

            # Prepare test data
            past_100_days = data_training[-100:]
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)  # FIXED append() warning
            input_data = scaler.fit_transform(np.array(final_df).reshape(-1, 1))

            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Make predictions
            y_predicted = model.predict(x_test)

            # Inverse transform predictions
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Plot 1: Closing Price vs Time (20 & 50 Days EMA)
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df.Close, 'y', label='Closing Price')
            ax1.plot(ema20, 'g', label='EMA 20')
            ax1.plot(ema50, 'r', label='EMA 50')
            ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price")
            ax1.legend()
            ema_chart_path = "static/ema_20_50.png"
            fig1.savefig(ema_chart_path)
            plt.close(fig1)

            # Plot 2: Closing Price vs Time (100 & 200 Days EMA)
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(df.Close, 'y', label='Closing Price')
            ax2.plot(ema100, 'g', label='EMA 100')
            ax2.plot(ema200, 'r', label='EMA 200')
            ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Price")
            ax2.legend()
            ema_chart_path_100_200 = "static/ema_100_200.png"
            fig2.savefig(ema_chart_path_100_200)
            plt.close(fig2)

            # Plot 3: Prediction vs Original Trend
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
            ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
            ax3.set_title("Prediction vs Original Trend")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Price")
            ax3.legend()
            prediction_chart_path = "static/stock_prediction.png"
            fig3.savefig(prediction_chart_path)
            plt.close(fig3)

            # Save dataset as CSV
            csv_file_path = f"static/{stock}_dataset.csv"
            df.to_csv(csv_file_path)

            return render_template('index.html', 
                                   plot_path_ema_20_50=ema_chart_path, 
                                   plot_path_ema_100_200=ema_chart_path_100_200, 
                                   plot_path_prediction=prediction_chart_path, 
                                   data_desc=data_desc.to_html(classes='table table-bordered'),
                                   dataset_link=csv_file_path)

        except Exception as e:
            return render_template("index.html", error=f"Error occurred: {str(e)}")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
