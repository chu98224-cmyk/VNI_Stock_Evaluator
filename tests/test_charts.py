"""
Unit tests for Charting Module
"""

import unittest
import pandas as pd
import numpy as np
from modules.charts import create_technical_chart, create_valuation_bands_chart, create_dupont_chart


class TestCharts(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range(start="2024-01-01", periods=60, freq="B")
        np.random.seed(42)
        prices = 25000 + np.cumsum(np.random.randn(60) * 200)
        self.df_price = pd.DataFrame({
            "time": dates,
            "open": prices - 100,
            "high": prices + 200,
            "low": prices - 200,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, size=60)
        })

    def test_technical_chart(self):
        fig = create_technical_chart(self.df_price, "HPG", show_bollinger=True, show_rsi=True, show_macd=True)
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.data), 5)

    def test_valuation_bands_chart(self):
        df_ratio = pd.DataFrame({
            "year": [2020, 2021, 2022, 2023, 2024],
            "quarter": [5, 5, 5, 5, 5],
            "pe": [8.5, 9.2, 7.8, 10.1, 8.9]
        })
        fig = create_valuation_bands_chart(df_ratio, 25000, metric_type="pe")
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.data), 5)


if __name__ == "__main__":
    unittest.main()
