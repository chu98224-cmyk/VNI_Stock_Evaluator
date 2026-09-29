"""
Unit tests for live features:
Order Flow Analysis, Technical Signals Speedometer, Trade Setup (R:R), Pivot Points, and Market Summary.
"""

import unittest
import pandas as pd
import numpy as np

from modules.live_analytics import (
    analyze_order_flow,
    calculate_technical_signals,
    calculate_trade_setup,
    calculate_pivot_points,
    generate_market_summary
)
from modules.charts import (
    create_technical_chart,
    create_intraday_vwap_chart,
    create_technical_gauge_chart,
    create_order_flow_donut_chart
)


class TestLiveFeatures(unittest.TestCase):

    def setUp(self):
        # Synthetic OHLCV history
        dates = pd.date_range(start='2026-01-01', periods=60, freq='D')
        prices = np.linspace(20000, 25000, 60) + np.random.normal(0, 200, 60)
        self.df_history = pd.DataFrame({
            'time': dates,
            'open': prices - 100,
            'high': prices + 250,
            'low': prices - 250,
            'close': prices,
            'volume': np.random.randint(100000, 2000000, 60)
        })

        # Synthetic Intraday ticks
        t_ticks = pd.date_range(start='2026-09-26 09:15:00', periods=50, freq='min')
        self.df_intraday = pd.DataFrame({
            'time': t_ticks,
            'price': np.linspace(24000, 24500, 50),
            'volume': [1000] * 50,
            'match_type': ['Buy'] * 35 + ['Sell'] * 15
        })

    def test_order_flow_analysis(self):
        res = analyze_order_flow(self.df_intraday, ref_price=24000)
        self.assertTrue(res['has_data'])
        self.assertGreater(res['buy_vol'], res['sell_vol'])
        self.assertAlmostEqual(res['buy_pct'], 70.0, delta=1.0)
        self.assertIn("Mua", res['pressure'])
        self.assertGreater(res['vwap_latest'], 0)

    def test_technical_signals(self):
        sig = calculate_technical_signals(self.df_history)
        self.assertIn(sig['consensus'], ["MUA MẠNH (STRONG BUY)", "MUA (BUY)", "THEO DÕI (NEUTRAL)", "BÁN (SELL)", "BÁN MẠNH (STRONG SELL)"])
        self.assertGreaterEqual(sig['score'], 0)
        self.assertLessEqual(sig['score'], 100)
        self.assertGreater(len(sig['signals']), 5)

    def test_trade_setup(self):
        setup = calculate_trade_setup(self.df_history)
        self.assertIn("entry_range", setup)
        self.assertIn("stop_loss", setup)
        self.assertIn("target_1", setup)
        self.assertIn("risk_reward_t1", setup)
        self.assertGreater(setup['target_1'], setup['current_price'])
        self.assertLess(setup['stop_loss'], setup['current_price'])

    def test_pivot_points(self):
        pivots = calculate_pivot_points(self.df_history)
        self.assertIn("classic", pivots)
        self.assertIn("fibonacci", pivots)
        self.assertGreater(pivots['classic']['R1'], pivots['classic']['PP'])
        self.assertLess(pivots['classic']['S1'], pivots['classic']['PP'])

    def test_charts_creation(self):
        vwap_fig = create_intraday_vwap_chart(self.df_intraday, "HPG", 24000)
        self.assertIsNotNone(vwap_fig)
        gauge_fig = create_technical_gauge_chart(75, "MUA MẠNH", 7, 2, 1)
        self.assertIsNotNone(gauge_fig)
        donut_fig = create_order_flow_donut_chart(70000, 30000)
        self.assertIsNotNone(donut_fig)
        
        # Test technical chart with pivot points
        pivots = calculate_pivot_points(self.df_history)
        tech_fig = create_technical_chart(
            df=self.df_history,
            symbol="HPG",
            show_pivots=True,
            pivots_data=pivots['classic']
        )
        self.assertIsNotNone(tech_fig)


if __name__ == '__main__':
    unittest.main()
