# Intraday Trading Prediction

This project provides a modular pipeline for training a three-layer bidirectional LSTM model that predicts the next ten 15-minute close prices for a given equity symbol. The model uses the most recent 200 candles and a set of technical indicators as inputs.

## Features

- Fetches 60 days of 15-minute OHLCV data via `yfinance`.
- Generates RSI, EMA20, EMA50, MACD (with signal & histogram), and change in returns.
- Builds lookback sequences of 200 intervals to forecast the next 10 close prices.
- Implements a deep bidirectional LSTM with three layers using PyTorch.
- Provides CLI utilities for daily training and intraday predictions.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model (intended once per day):

```bash
python main.py train --ticker AAPL --epochs 30
```

Generate the next 10 interval predictions after training:

```bash
python main.py predict --ticker AAPL
```

Daily training artifacts (model weights and scaler) are stored in the `models/` directory by default.
