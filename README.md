# 🇻🇳 VNI Stock Evaluator & Live Market Board (Định Giá & Bảng Điện Tử Trực Tuyến)

An interactive, all-in-one **Vietnamese Stock Evaluation, Real-Time Market Board, Order Flow & Technical Analysis Dashboard** built with **100% Python**, **Streamlit**, and **Plotly**, powered by **vnstock 4.0.8**.

---

## 🌟 Key Features

### 1. 🖥️ VCBS-Style Electronic Live Price Board (Bảng Giá Trực Tuyến)
- **Top 4 Real-time Indices (VN-INDEX, VN30, HNX-INDEX, UPCOM)**: Live points, point change (+/- and %), volume (Tr), and turnover value (K Tỷ).
- **Exchange & Sector Switcher**: One-click filtering across **HOSE, VN30, Banks, Tech, Steel, Real Estate, Securities, Consumer, Energy...**
- **Full 3-Level Depth Order Book**:
  - `TC` (Tham chiếu - Vàng), `Trần` (Tím), `Sàn` (Xanh lơ)
  - `Dư Mua`: G3, KL3, G2, KL2, G1, KL1
  - `Khớp Lệnh`: Giá khớp, +/-, %, KL khớp
  - `Dư Bán`: G1, KL1, G2, KL2, G3, KL3
  - `Tổng KL`, `Cao`, `Thấp`, `TB`, `Khối Ngoại Mua / Bán`
- **Smart Market Wrap (Bản Tin Tổng Hợp)**: Automated market commentary with breadth (Tăng/Trần, Giảm/Sàn, TC), top gainers, losers, and liquidity leaders.

### 2. ⚡ Live Session Order Flow & VWAP (Khớp Lệnh Trong Phiên)
- **Active Buy vs Sell Volume Breakdown**: Real-time ratio of market orders hitting Ask (*Active Buy*) vs Bid (*Active Sell*).
- **Intraday VWAP (Volume-Weighted Average Price)**: Interactive intraday chart comparing execution price vs VWAP baseline.
- **Liquidity Pressure Meter & Donut Distribution**: Identifies whether bulls or bears are accumulating positions.

### 3. 🎯 Quantitative Buy/Sell Decision Suite & Trade Setup
- **TradingView-Style Speedometer Consensus**: 15-factor quantitative scoring (RSI, Stochastic, MACD, Bollinger Bands, EMA 9/21, MA20/50/200, CCI).
- **Automated Trade Setup (Kế Hoạch Giao Dịch)**:
  - Recommended Entry Zone corridor.
  - Take Profit Target 1 (TP1) & Target 2 (TP2).
  - ATR-based Dynamic Stop-Loss level.
  - Calculated **Risk-Reward (R:R) Ratio** (e.g. 1 : 2.0).
- **Pivot Points Matrix**: Daily Classic and Fibonacci Support & Resistance levels (S3, S2, S1, PP, R1, R2, R3).

### 4. 📈 Interactive Technical Analysis & Bollinger Bands
- **TradingView 60fps Lightweight Engine** + Multi-subplot Plotly Candlestick view.
- Bollinger Bands (20, 2), Moving Averages (MA20/50/200), RSI (14), and MACD (12, 26, 9).

### 5. 💰 Multi-Model Intrinsic Valuation Suite
- 5-Year Historical P/E & P/B Multiple Corridor Bands.
- Benjamin Graham Number & Bond-adjusted valuation.
- DCF / FCFE with 2D Sensitivity Matrix.

### 6. 🏥 Financial Quality & Health Scoring
- Piotroski F-Score (0–9), Altman Z''-Score solvency risk, DuPont 3-Step ROE decomposition, and Cash Conversion Cycle.

### 7. 🔍 Sector Peer Screener
- Real-time peer scanner across 10 Vietnamese industry sectors.

---

## 🛠️ Tech Stack & Requirements

- **Python**: `>= 3.10`
- **Data Engine**: `vnstock >= 4.0.8` (VCI / KBS broker connectors)
- **Frontend / UI**: `Streamlit >= 1.35.0`
- **Charting**: `Plotly >= 5.18.0`
- **Analytics**: `pandas`, `numpy`, `scipy`

---

## 🚀 Quick Start Guide

### 1. Clone the repository
```bash
git clone https://github.com/chu98224-cmyk/VNI_Stock_Evaluator.git
cd VNI_Stock_Evaluator
```

### 2. Install Dependencies
```bash
py -3.10 -m pip install -r requirements.txt
```

### 3. Launch the Application
```bash
py -3.10 -m streamlit run app.py
```
The app will open automatically in your default browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
VNI_Stock_Evaluator/
├── app.py                     # Main Streamlit Dashboard application
├── requirements.txt           # Project dependencies (Python >= 3.10)
├── .gitignore                 # Git ignore configuration
├── README.md                  # Documentation & user guide
│
├── modules/
│   ├── __init__.py
│   ├── data_fetcher.py        # vnstock connector with caching (@st.cache_data)
│   ├── valuation_engine.py    # DCF, Graham, Multiple Bands, DDM, Sensitivity Matrix
│   ├── scoring_engine.py      # Piotroski F-Score (0-9), Altman Z-Score, DuPont breakdown
│   └── charts.py              # Interactive Plotly candlestick & financial charts
│
└── tests/
    ├── test_valuation.py      # Unit tests for valuation algorithms
    ├── test_scoring.py        # Unit tests for financial health scoring
    └── test_charts.py         # Unit tests for technical chart generation
```

---

## 🧪 Running Unit Tests

Run the test suite to verify all valuation and scoring modules:
```bash
py -3.10 -m unittest discover -s tests -p "test_*.py"
```

---

## 🗺️ Next Steps & Roadmap

- [ ] **One-Click PDF / Excel Valuation Reports**: Export full investment memos with DCF schedules.
- [ ] **Forensic Accounting & Red Flag Detector**: Accrual anomaly detection and receivable spikes.
- [ ] **Watchlist & Price Alerts**: Track custom target buy/sell zones based on Margin of Safety.
- [ ] **Macro Barometer**: Foreign investor flow (*Khối ngoại*) & VN-Index market breadth.

---

## 📄 License
MIT License. Data provided for educational and analytical purposes via public market endpoints.
