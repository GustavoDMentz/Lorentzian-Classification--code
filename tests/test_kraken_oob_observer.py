import unittest
from research.kraken_oob_observer import summarize


class ObserverTests(unittest.TestCase):
    def test_spread_and_imbalance(self):
        book = {"bids": [["99", "2", "0"]], "asks": [["101", "1", "0"]]}
        ticker = {"c": ["100", "1"], "v": ["10", "20"]}
        row = summarize("OOBUSD", book, ticker, "2026-10-23T00:00:00+00:00")
        self.assertAlmostEqual(row["spread_pct"], 2)
        self.assertAlmostEqual(row["bid_depth_10"], 198)
        self.assertAlmostEqual(row["ask_depth_10"], 101)
        self.assertAlmostEqual(row["imbalance_10"], 97/299)

    def test_reject_crossed_book(self):
        with self.assertRaises(ValueError):
            summarize("OOBUSD", {"bids": [["101", "1"]], "asks": [["100", "1"]]},
                      {"c": ["100"], "v": ["1", "1"]}, "now")


if __name__ == "__main__":
    unittest.main()
