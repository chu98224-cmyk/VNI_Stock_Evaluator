"""
Unit tests for Valuation Engine
"""

import unittest
import pandas as pd
from modules.valuation_engine import (
    calculate_pe_pb_bands,
    calculate_graham_valuation,
    calculate_dcf,
    generate_dcf_sensitivity_matrix,
    calculate_ddm,
    synthesize_valuations
)


class TestValuationEngine(unittest.TestCase):

    def test_graham_valuation(self):
        eps = 3000.0
        bvps = 25000.0
        res = calculate_graham_valuation(eps, bvps, growth_rate=8.0, bond_yield=3.2)
        self.assertIsNotNone(res["graham_number"])
        self.assertGreater(res["graham_number"], 0)
        self.assertIsNotNone(res["revised_graham"])
        self.assertGreater(res["revised_graham"], 0)

    def test_dcf_valuation(self):
        fcf = 5000000000000.0  # 5,000 billion VND
        growth = 10.0
        term_growth = 3.0
        wacc = 11.0
        shares = 1000000000.0  # 1 billion shares
        res = calculate_dcf(fcf, growth, term_growth, wacc, shares)
        self.assertIsNotNone(res)
        self.assertGreater(res["fair_value_per_share"], 0)
        self.assertEqual(len(res["projections"]), 5)

    def test_dcf_sensitivity_matrix(self):
        fcf = 1000000000000.0
        matrix = generate_dcf_sensitivity_matrix(fcf, 10.0, 11.0, 3.0, 500000000.0)
        self.assertFalse(matrix.empty)
        self.assertEqual(matrix.shape[0], 5)

    def test_pe_pb_bands(self):
        df_ratio = pd.DataFrame({
            "pe": [8.0, 9.5, 11.0, 10.0, 12.0, 9.0],
            "pb": [1.2, 1.4, 1.6, 1.5, 1.8, 1.3]
        })
        bands = calculate_pe_pb_bands(df_ratio, current_eps=2500, current_bvps=20000, current_price=25000)
        self.assertIsNotNone(bands["pe_fair_value"])
        self.assertIsNotNone(bands["pb_fair_value"])
        self.assertIn("Median", bands["pe_bands"])

    def test_synthesize_valuations(self):
        synth = synthesize_valuations(
            current_price=25000,
            pe_fair=30000,
            pb_fair=28000,
            graham_number=29000,
            dcf_fair=32000
        )
        self.assertGreater(synth["blended_fair_value"], 25000)
        self.assertGreater(synth["margin_of_safety_pct"], 0)
        self.assertIn("Undervalued", synth["status"])


if __name__ == "__main__":
    unittest.main()
