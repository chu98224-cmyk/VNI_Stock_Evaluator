"""
Unit tests for Financial Scoring Engine
"""

import unittest
import pandas as pd
from modules.scoring_engine import (
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dupont_analysis
)


class TestScoringEngine(unittest.TestCase):

    def setUp(self):
        self.df_ratio = pd.DataFrame([
            {
                "year": 2023, "quarter": 4, "roa": 8.0, "roe": 18.0,
                "after_tax_profit_margin": 12.0, "financial_leverage": 1.5,
                "debt_to_equity": 0.8, "current_ratio": 1.8, "gross_margin": 22.0,
                "asset_turnover": 0.9, "number_of_shares_mkt_cap": 1000000
            },
            {
                "year": 2024, "quarter": 4, "roa": 10.0, "roe": 21.0,
                "after_tax_profit_margin": 14.0, "financial_leverage": 1.4,
                "debt_to_equity": 0.7, "current_ratio": 2.1, "gross_margin": 24.0,
                "asset_turnover": 1.0, "number_of_shares_mkt_cap": 1000000
            }
        ])

    def test_piotroski_f_score(self):
        res = calculate_piotroski_f_score(self.df_ratio)
        self.assertGreaterEqual(res["total_score"], 7)
        self.assertEqual(res["max_score"], 9)
        self.assertEqual(len(res["details"]), 9)

    def test_altman_z_score(self):
        res = calculate_altman_z_score(self.df_ratio)
        self.assertIsNotNone(res["z_score"])
        self.assertIn("Safe", res["zone"])

    def test_dupont_analysis(self):
        dupont = calculate_dupont_analysis(self.df_ratio)
        self.assertEqual(len(dupont), 2)
        self.assertIn("roe", dupont[0])
        self.assertIn("net_margin_pct", dupont[0])


if __name__ == "__main__":
    unittest.main()
