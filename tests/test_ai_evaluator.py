import unittest
from modules.ai_evaluator import (
    extract_stock_symbol_from_prompt,
    extract_target_price_from_prompt,
    evaluate_stock_quantitative,
    evaluate_stock_ai
)


class TestAiEvaluator(unittest.TestCase):

    def test_extract_symbol(self):
        self.assertEqual(extract_stock_symbol_from_prompt("Định giá HPG ở vùng này thế nào"), "HPG")
        self.assertEqual(extract_stock_symbol_from_prompt("Mã SSI có điểm mua không?", "VNM"), "SSI")
        self.assertEqual(extract_stock_symbol_from_prompt("Cổ phiếu FPT định giá bao nhiêu?"), "FPT")
        self.assertEqual(extract_stock_symbol_from_prompt("Hỏi về thị trường chung", "MWG"), "MWG")

    def test_extract_target_price(self):
        self.assertEqual(extract_target_price_from_prompt("HPG giá 28k có mua được không"), 28000.0)
        self.assertEqual(extract_target_price_from_prompt("VNM ở 65.5k có nên bắt đáy"), 65500.0)
        self.assertEqual(extract_target_price_from_prompt("FPT giá 135,000 có đắt không"), 135000.0)
        self.assertIsNone(extract_target_price_from_prompt("Đánh giá tổng quan doanh nghiệp"))

    def test_evaluate_stock_quantitative(self):
        dummy_dossier = {
            "symbol": "HPG",
            "company_name": "Tập đoàn Hòa Phát",
            "sector": "Thép & Vật liệu xây dựng",
            "exchange": "HOSE",
            "current_market_price": 28000,
            "evaluated_price": 28000,
            "is_custom_eval_price": False,
            "fundamental_metrics": {
                "pe": 10.5,
                "pb": 1.4,
                "roe": 15.2,
                "dividend_yield": 4.5,
                "debt_to_equity": 0.65,
                "eps": 2666,
                "bvps": 20000
            },
            "valuation": {
                "blended_fair_value": 35000,
                "margin_of_safety_pct": 25.0,
                "status": "Significantly Undervalued",
                "pe_median_fair": 34000,
                "pb_median_fair": 36000,
                "graham_number": 34641,
                "dcf_fair": 36000,
                "ddm_fair": 32000
            },
            "financial_health": {
                "piotroski_score": 7,
                "piotroski_max": 9,
                "health_status": "Strong Health",
                "altman_z_score": 3.1,
                "altman_zone": "Safe Zone"
            },
            "technicals": {
                "score": 75,
                "consensus": "BUY",
                "rsi": 56.4,
                "macd": 0.35,
                "trade_setup": {
                    "entry_range": (27500, 28200),
                    "target_1": 31000,
                    "target_1_pct": 10.7,
                    "target_2": 33500,
                    "target_2_pct": 19.6,
                    "stop_loss": 26200,
                    "stop_loss_pct": -6.4,
                    "risk_reward_t1": 1.67,
                    "risk_reward_t2": 3.06
                },
                "pivots": {"PP": 28000, "R1": 29000, "R2": 30500, "S1": 27200, "S2": 26500}
            }
        }
        
        report = evaluate_stock_quantitative(dummy_dossier, "Có nên mua HPG ở giá 28k không?")
        self.assertIn("HPG", report)
        self.assertIn("BÁO CÁO ĐÁNH GIÁ ĐẦU TƯ AI", report)
        self.assertIn("KHUYẾN NGHỊ", report)
        self.assertIn("VALUATION", report)
        self.assertIn("FINANCIAL QUALITY", report)
        self.assertIn("Piotroski", report)

    def test_evaluate_stock_ai_builtin_fallback(self):
        dummy_dossier = {
            "symbol": "VNM",
            "company_name": "Vinamilk",
            "current_market_price": 68000,
            "evaluated_price": 68000,
            "valuation": {"blended_fair_value": 75000, "margin_of_safety_pct": 10.3},
            "financial_health": {"piotroski_score": 8, "health_status": "Strong"},
            "technicals": {"score": 60, "consensus": "NEUTRAL", "rsi": 48.0}
        }
        res = evaluate_stock_ai(dummy_dossier, "Đánh giá VNM", provider="builtin")
        self.assertIn("VNM", res)

    def test_evaluate_stock_quantitative_with_all_none(self):
        # Edge case: All values None / missing
        none_dossier = {
            "symbol": "ABC",
            "company_name": None,
            "sector": None,
            "current_market_price": None,
            "evaluated_price": None,
            "is_custom_eval_price": False,
            "fundamental_metrics": {
                "pe": None, "pb": None, "roe": None, "dividend_yield": None, "debt_to_equity": None
            },
            "valuation": {
                "blended_fair_value": None, "margin_of_safety_pct": None, "status": None,
                "pe_median_fair": None, "pb_median_fair": None, "graham_number": None,
                "revised_graham": None, "dcf_fair": None, "ddm_fair": None
            },
            "financial_health": {
                "piotroski_score": None, "health_status": None, "altman_z_score": None, "altman_zone": None
            },
            "technicals": {
                "score": None, "consensus": None, "rsi": None, "macd": None,
                "trade_setup": None, "pivots": None
            }
        }
        res = evaluate_stock_quantitative(none_dossier, "Đánh giá ABC")
        self.assertIn("ABC", res)
        self.assertIn("BÁO CÁO ĐÁNH GIÁ ĐẦU TƯ AI", res)


if __name__ == "__main__":
    unittest.main()
