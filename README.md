# Hybrid Quant Fund — Research Prototype

> **Research and educational prototype only.**
> This project is **not** a live trading system.
> It is **not** investment advice.
> It is **not** a regulated investment product or fund.
> All results are hypothetical simulations on historical data.
> Past simulated performance does not guarantee future results.

---

## Overview

`hybrid-quant-fund` is a research prototype exploring a hybrid quantitative architecture that combines:

* financial feature engineering;
* a Random Forest decision layer;
* market-derived contextual proxy signals;
* curated-source NLP signal extraction;
* strict train / validation / test separation;
* transaction-cost-aware backtesting;
* model governance and diagnostic tools.

The goal is **not** to maximise backtest performance.
The goal is to study whether contextual information can improve model robustness across different market regimes without introducing data leakage or overfitting.

---

## Core Research Question

Can a controlled contextual signal layer improve a classical quantitative model?

This project tests that question progressively:

1. Start with price-based financial features.
2. Add market-derived contextual proxy signals.
3. Add diagnostic tools to test whether those signals genuinely help.
4. Add a curated-source NLP layer that converts text into numerical features.
5. Keep the Random Forest as the sole decision-making model.

The NLP / LLM layer is **not a trader**.
It does **not** make buy, sell, or hold decisions.
It only converts selected textual sources into structured numerical features.

---

## Version History

| Version |    Status | Description                                                                                                                      |
| ------- | --------: | -------------------------------------------------------------------------------------------------------------------------------- |
| V1      | Completed | Neutral mock contextual layer. All `ai_*` signals fixed at 0.0. Used to validate the pipeline.                                   |
| V2      | Completed | Market-derived proxy signals based on SPY/TLT momentum, relative ETF momentum, and volatility regime. No NLP.                    |
| V2.1    | Completed | Research diagnostics: ablation testing, rolling diagnostics, permutation importance, signal correlations, and regime analysis.   |
| V3      |   Current | Curated-source NLP layer. Local text sources are transformed into validated `nlp_*` signals, then blended with V2 proxy signals. |

---

## Architecture

```text
hybrid-quant-fund/
├── data/
│   ├── prices.csv
│   └── text_sources/
│       ├── 2020-03-15_covid_market_stress.txt
│       ├── 2022-06-15_fed_hawkish_inflation.txt
│       └── 2023-01-20_recovery_disinflation.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── ai_signal.py
│   ├── nlp_schema.py
│   ├── nlp_extractor.py
│   ├── nlp_signal.py
│   ├── text_sources.py
│   ├── model.py
│   ├── backtest.py
│   ├── risk.py
│   ├── ablation.py
│   ├── diagnostics.py
│   ├── regime.py
│   ├── governance.py
│   └── app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## V3 Pipeline

```text
Curated local text sources
        ↓
text_sources.py
        ↓
nlp_extractor.py
        ↓
nlp_schema.py validation
        ↓
nlp_signal.py daily alignment
        ↓
ai_signal.py blends NLP + V2 proxy signals
        ↓
Random Forest classifier
        ↓
Backtest with transaction costs
        ↓
Streamlit diagnostics dashboard
```

The V3 design is intentionally conservative:

* no random browsing;
* no live trading;
* no broker integration;
* no leverage;
* no short selling;
* no crypto;
* no API requirement by default;
* no hyperparameter tuning based on the test period.

---

## Data Sources

### Market Data

ETF price data is downloaded using `yfinance` and cached locally.

The default ETF universe is:

| Ticker | Description                          |
| ------ | ------------------------------------ |
| SPY    | S&P 500 ETF                          |
| QQQ    | Nasdaq-100 ETF                       |
| TLT    | 20+ Year US Treasury Bond ETF        |
| IEF    | 7–10 Year US Treasury Bond ETF       |
| GLD    | Gold ETF                             |
| EFA    | Developed international equities ETF |

### Text Data

V3 reads curated local source files from:

```text
data/text_sources/
```

Supported formats:

* `.txt`
* `.md`
* `.csv`

The provided example files are synthetic examples for testing only:

```text
2020-03-15_covid_market_stress.txt
2022-06-15_fed_hawkish_inflation.txt
2023-01-20_recovery_disinflation.txt
```

They are **not real market documents**.

---

## Temporal Split

The project uses strict temporal separation:

| Period     | Dates                   | Purpose                         |
| ---------- | ----------------------- | ------------------------------- |
| Training   | 2010-01-01 → 2018-12-31 | Model fitting only              |
| Validation | 2019-01-01 → 2020-12-31 | Out-of-sample diagnostic period |
| Test       | 2021-01-01 → 2023-12-31 | Final held-out evaluation       |

Rules:

* no hyperparameter tuning after validation;
* no test-period tuning;
* no future source document can influence past features;
* all rolling features use trailing windows only;
* any structural change requires full retraining.

---

## Feature Engineering

### Financial Features

| Feature    | Description                                       |
| ---------- | ------------------------------------------------- |
| `mom_5d`   | 5-day log-return momentum                         |
| `mom_10d`  | 10-day log-return momentum                        |
| `mom_21d`  | 21-day log-return momentum                        |
| `mom_63d`  | 63-day log-return momentum                        |
| `vol_21d`  | 21-day realised volatility                        |
| `ma_ratio` | 10-day moving average / 63-day moving average − 1 |
| `drawdown` | Price / 63-day rolling high − 1                   |

### V2 Contextual Proxy Signals

| Feature                | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `ai_macro_sentiment`   | SPY momentum minus TLT momentum, rolling z-scored              |
| `ai_sector_momentum`   | Ticker momentum relative to equal-weight universe momentum     |
| `ai_volatility_regime` | Universe volatility regime, negated so high stress is negative |

### V3 NLP Signals

| Feature                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `nlp_macro_sentiment`    | Extracted macro tone from curated text             |
| `nlp_sector_sentiment`   | Extracted sector-specific tone                     |
| `nlp_volatility_risk`    | Extracted volatility / stress risk                 |
| `nlp_inflation_pressure` | Extracted inflation pressure                       |
| `nlp_rates_pressure`     | Extracted rate-hike / monetary tightening pressure |
| `nlp_recession_risk`     | Extracted recession risk                           |
| `nlp_confidence`         | Confidence score from extraction                   |

The existing `ai_*` columns are retained and enriched by blending:

```text
V2 price-proxy context + V3 NLP-derived context
```

This preserves backward compatibility with earlier versions.

---

## NLP / LLM Layer

V3 supports two modes.

### 1. Heuristic Mode

This mode works offline and requires no API key.

It uses transparent keyword-based scoring to convert curated text documents into numerical NLP features.

Example concepts:

* inflation;
* rate hikes;
* recession risk;
* volatility stress;
* recovery;
* risk-on sentiment;
* defensive market tone.

### 2. Optional LLM Mode

The project architecture supports an optional LLM extraction path, but the app does not require it by default.

The LLM layer must:

* read only provided local source documents;
* return strict JSON;
* not browse the web;
* not use model memory as a market data source;
* not make trading recommendations;
* not decide positions;
* fall back safely if unavailable or invalid.

---

## Model

The decision layer is a `RandomForestClassifier`.

The Random Forest receives numerical features and predicts the probability that each ETF will have a positive next-day return.

Important:

* the Random Forest is the sole decision-making layer;
* the NLP / LLM layer only creates features;
* the model is long-only;
* there is no leverage;
* there is no short selling;
* there is no live execution.

---

## Backtest Rules

The backtest engine applies the following rules:

* each day, hold ETFs whose predicted probability is above the configured signal threshold;
* equal-weight all selected ETFs;
* hold cash if no ETF qualifies;
* use yesterday’s signal for today’s position;
* apply transaction costs based on daily turnover;
* compare against an equal-weight ETF benchmark.

Transaction costs are configured in `config.py`.

---

## Diagnostics

V2.1 and V3 add research diagnostics to avoid relying only on headline returns.

### Ablation Test

Compares:

| Model           | Description                                          |
| --------------- | ---------------------------------------------------- |
| Baseline        | Financial features only                              |
| V2 Proxy        | Financial features + V2 market-derived proxy signals |
| V3 NLP-Enriched | Financial features + V2 proxy + V3 NLP signals       |

This tests whether contextual features actually improve performance.

### Permutation Importance

Measures whether the trained model depends on specific features in:

* training;
* validation;
* test.

### Signal Correlations

Measures correlation between contextual signals and future returns over:

* next 1 day;
* next 5 days;
* next 21 days.

### Rolling Diagnostics

Shows rolling performance stability:

* rolling return;
* rolling volatility;
* rolling Sharpe ratio.

### Regime Analysis

Classifies market periods into regimes such as:

* Risk-On Trend;
* Defensive / Risk-Off;
* Bond-Led Defensive;
* High-Vol Stress;
* Neutral / Mixed.

This helps evaluate whether the model works only in certain regimes.

---

## Streamlit Dashboard

Run the dashboard with:

```zsh
streamlit run src/app.py
```

Main tabs include:

* Training;
* Validation;
* Test;
* Ablation Test;
* Rolling Diagnostics;
* Regime Analysis;
* NLP / LLM Signals;
* Context Signals;
* Feature Analysis;
* Governance.

---

## Installation

```zsh
git clone <your-repository-url>
cd hybrid-quant-fund

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data/text_sources

streamlit run src/app.py
```

---

## Running Checks

```zsh
source venv/bin/activate
python -m py_compile src/*.py
streamlit cache clear
streamlit run src/app.py
```

---

## Example Text Sources

Place curated `.txt`, `.md`, or `.csv` files in:

```text
data/text_sources/
```

Example filename format:

```text
YYYY-MM-DD_short_description.txt
```

Example:

```text
2022-06-15_fed_hawkish_inflation.txt
```

The date controls when the NLP signal becomes available.
A document dated after a market date must not affect earlier features.

---

## Governance Principles

This project follows strict research governance:

1. Training data is strictly bounded to 2010–2018.
2. Validation data is used for diagnostics only.
3. Test data is held out for final evaluation.
4. No hyperparameters are tuned after viewing validation.
5. No test-period tuning is allowed.
6. All rolling features are trailing-only.
7. Text source dates control signal availability.
8. NLP signals are validated and clipped before entering the model.
9. The NLP / LLM layer does not make trading decisions.
10. The Random Forest remains the sole decision-making layer.
11. The system is not a live trading system.
12. The system is not investment advice.

---

## Limitations

* Small ETF universe.
* Synthetic example text sources are not real market documents.
* Heuristic NLP is simple and keyword-based.
* Optional LLM extraction requires careful governance and validation.
* Backtest results are highly sensitive to period choice.
* Transaction costs are simplified.
* Benchmark is frictionless.
* No market impact model.
* No live trading.
* No production risk controls.
* No statistical significance testing yet.

---

## Future Work

Possible future improvements:

* replace synthetic examples with real curated source documents;
* add proper source metadata and provenance tracking;
* add bootstrapped confidence intervals;
* add walk-forward validation;
* add larger ETF universe;
* add real FinBERT or LLM extraction mode;
* add model cards and experiment registry;
* add stronger leakage tests;
* add source-level audit trail.

---

## Disclaimer

This repository is for research and educational purposes only.

It is not financial advice.
It is not a live trading system.
It is not a regulated investment product.
It is not suitable for real-money trading.

Do not make investment decisions based on this project.
