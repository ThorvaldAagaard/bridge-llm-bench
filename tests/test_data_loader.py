"""
Unit tests for data loading functionality.

Tests cover both text and numeric format loading, hand decoding, and edge cases.
"""

import unittest
import tempfile
import os
from pathlib import Path

from bridge_llm_bench.parsers.data_loader import (
    load_dataset,
    ensure_default_dataset,
    _load_text_format,
    _load_numeric_format,
    _decode_hand,
    _id2card,
    _next_int_line
)


class TestDataLoading(unittest.TestCase):
    """Test suite for data loading functions."""
    
    def test_text_format_loading(self):
        """Test loading Bridge data from text format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("S:AKQ H:987 D:AK C:5432 | 1H Pass | 2H\n")
            f.write("S:J987 H:AK D:Q987 C:43 | 1NT Pass | 2C\n")
            f.write("S:32 H:AKQJ D:KQ C:AKQ2 | | 2NT\n")  # Empty auction
            f.write("Invalid line without pipes\n")  # Should be skipped
            f.write("S:AK H:AK D:AK C:AK9876 | 2C | 2D\n")
            temp_path = f.name
        
        try:
            # Test with limit
            records = load_dataset(temp_path, n_records=2)
            self.assertEqual(len(records), 2)
            
            # Check first record
            hand, auction, bid = records[0]
            self.assertEqual(hand, "S:AKQ H:987 D:AK C:5432")
            self.assertEqual(auction, "1H Pass")
            self.assertEqual(bid, "2H")
            
            # Check second record
            hand, auction, bid = records[1]
            self.assertEqual(hand, "S:J987 H:AK D:Q987 C:43")
            self.assertEqual(auction, "1NT Pass")
            self.assertEqual(bid, "2C")
            
            # Test loading all records
            all_records = load_dataset(temp_path)
            self.assertEqual(len(all_records), 4)  # Invalid line skipped
            
            # Check empty auction case
            hand, auction, bid = all_records[2]
            self.assertEqual(auction, "")
            self.assertEqual(bid, "2NT")
            
        finally:
            os.unlink(temp_path)
    
    def test_numeric_format_4x13_loading(self):
        """Test loading Bridge data from numeric format (4 lines × 13 cards)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # South hand (13 cards)
            f.write("0 1 2 3 4 5 6 7 8 9 10 11 12\n")
            # North hand (ignored)
            f.write("13 14 15 16 17 18 19 20 21 22 23 24 25\n")
            # East hand (ignored)
            f.write("26 27 28 29 30 31 32 33 34 35 36 37 38\n")
            # West hand (ignored)
            f.write("39 40 41 42 43 44 45 46 47 48 49 50 51\n")
            # Auction and answer: 1C Pass 1D
            f.write("55 52 56\n")
            temp_path = f.name
        
        try:
            records = load_dataset(temp_path, n_records=1)
            self.assertEqual(len(records), 1)
            
            hand, auction, bid = records[0]
            
            # Verify hand contains all suits
            self.assertIn("S:", hand)
            self.assertIn("H:", hand)
            self.assertIn("D:", hand)
            self.assertIn("C:", hand)
            
            # Verify auction
            self.assertEqual(auction, "1C Pass")
            
            # Verify bid
            self.assertEqual(bid, "1D")
            
        finally:
            os.unlink(temp_path)
    
    def test_numeric_format_1x52_loading(self):
        """Test loading Bridge data from numeric format (1 line × 52+ cards)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # 52 cards + auction IDs
            cards = list(range(52))
            auction_ids = [55, 52, 56]  # 1C Pass 1D
            f.write(" ".join(map(str, cards + auction_ids)) + "\n")
            temp_path = f.name
        
        try:
            records = load_dataset(temp_path, n_records=1)
            self.assertEqual(len(records), 1)
            
            hand, auction, bid = records[0]
            
            # South gets first 13 cards (0-12)
            # These should be clubs 2-A
            self.assertIn("C:23456789TJQKA", hand)
            
        finally:
            os.unlink(temp_path)
    
    def test_card_id_to_string_conversion(self):
        """Test conversion of card IDs to string representation."""
        test_cases = [
            # Clubs (0-12)
            (0, "C2"), (1, "C3"), (2, "C4"), (3, "C5"), (4, "C6"),
            (5, "C7"), (6, "C8"), (7, "C9"), (8, "CT"), (9, "CJ"),
            (10, "CQ"), (11, "CK"), (12, "CA"),
            
            # Diamonds (13-25)
            (13, "D2"), (14, "D3"), (25, "DA"),
            
            # Hearts (26-38)
            (26, "H2"), (38, "HA"),
            
            # Spades (39-51)
            (39, "S2"), (51, "SA"),
        ]
        
        for card_id, expected in test_cases:
            with self.subTest(card_id=card_id):
                result = _id2card(card_id)
                self.assertEqual(result, expected)
    
    def test_hand_decoding(self):
        """Test decoding of card IDs to hand string."""
        # Test with a specific hand
        # SA, SK, SQ, HA, HK, DA, DK, CA, CK, CQ, CJ, C3, C2
        card_ids = [51, 50, 49, 38, 37, 25, 24, 12, 11, 10, 9, 1, 0]
        
        hand = _decode_hand(card_ids)
        
        # Check format
        self.assertRegex(hand, r"S:\S+ H:\S+ D:\S+ C:\S+")
        
        # Check specific cards are in correct suits
        parts = dict(part.split(":") for part in hand.split())
        
        # Spades should have A, K, Q
        self.assertIn("A", parts["S"])
        self.assertIn("K", parts["S"])
        self.assertIn("Q", parts["S"])
        
        # Hearts should have A, K
        self.assertIn("A", parts["H"])
        self.assertIn("K", parts["H"])
        
        # Diamonds should have A, K
        self.assertIn("A", parts["D"])
        self.assertIn("K", parts["D"])
        
        # Clubs should have A, K, Q, J, 3, 2
        self.assertIn("A", parts["C"])
        self.assertIn("K", parts["C"])
        self.assertIn("Q", parts["C"])
        self.assertIn("J", parts["C"])
        self.assertIn("3", parts["C"])
        self.assertIn("2", parts["C"])
    
    def test_ensure_default_dataset(self):
        """Test default dataset download functionality."""
        # Test with existing file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            result = ensure_default_dataset(temp_path)
            self.assertEqual(result, temp_path)
            self.assertTrue(os.path.exists(temp_path))
        finally:
            os.unlink(temp_path)
        
        # Test download would be triggered for non-existent file
        # (We won't actually download in tests)
        non_existent = Path("test_data/test_dataset.txt")
        self.assertFalse(non_existent.exists())
    
    def test_mixed_format_handling(self):
        """Test handling of files with mixed or invalid formats."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Empty lines
            f.write("\n\n")
            # Invalid format
            f.write("Some random text\n")
            # Valid text format
            f.write("S:AKQ H:987 D:AK C:5432 | 1H Pass | 2H\n")
            # More invalid
            f.write("12 34 56\n")  # Too few numbers for numeric format
            temp_path = f.name
        
        try:
            records = load_dataset(temp_path)
            # Should only load the one valid text format line
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0][2], "2H")
        finally:
            os.unlink(temp_path)
    
    def test_edge_cases_in_auction_parsing(self):
        """Test edge cases in auction sequence parsing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Multiple spaces in auction
            f.write("S:AKQ H:987 D:AK C:5432 | 1H  Pass   2H | 3H\n")
            # Auction with only one bid
            f.write("S:AKQ H:987 D:AK C:5432 | 1NT | 3NT\n")
            # Very long auction
            f.write("S:AKQ H:987 D:AK C:5432 | Pass Pass 1C Pass 1H Pass 2H Pass 4H Pass Pass Pass | Pass\n")
            temp_path = f.name
        
        try:
            records = load_dataset(temp_path)
            self.assertEqual(len(records), 3)
            
            # Check multiple spaces handled correctly
            _, auction1, _ = records[0]
            self.assertEqual(auction1, "1H  Pass   2H")
            
            # Check single bid auction
            _, auction2, _ = records[1]
            self.assertEqual(auction2, "1NT")
            
            # Check long auction
            _, auction3, _ = records[2]
            self.assertTrue(auction3.startswith("Pass Pass 1C"))
            
        finally:
            os.unlink(temp_path)
    
    def test_next_int_line_helper(self):
        """Test the _next_int_line helper function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1 2 3\n")
            f.write("\n")  # Empty line
            f.write("  \n")  # Whitespace line
            f.write("4 5 6\n")
            f.write("not numbers\n")  # This will cause an error if read
            temp_path = f.name
        
        try:
            with open(temp_path, 'r') as fh:
                # First call should return [1, 2, 3]
                result1 = _next_int_line(fh)
                self.assertEqual(result1, [1, 2, 3])
                
                # Second call should skip empty lines and return [4, 5, 6]
                result2 = _next_int_line(fh)
                self.assertEqual(result2, [4, 5, 6])
                
                # Reading the "not numbers" line would raise ValueError
                # but we won't test that here as it would crash
                
        finally:
            os.unlink(temp_path)


class TestDatasetAutoDetection(unittest.TestCase):
    """Test suite for automatic dataset format detection."""
    
    def test_format_detection_text(self):
        """Test automatic detection of text format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Start with some empty lines
            f.write("\n\n")
            # Then a text format line
            f.write("S:AKQ H:987 D:AK C:5432 | 1H Pass | 2H\n")
            temp_path = f.name
        
        try:
            # Should detect as text format due to pipe character
            records = load_dataset(temp_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0][2], "2H")
        finally:
            os.unlink(temp_path)
    
    def test_format_detection_numeric(self):
        """Test automatic detection of numeric format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Start with some empty lines
            f.write("\n\n")
            # Then numeric format (no pipes)
            f.write("0 1 2 3 4 5 6 7 8 9 10 11 12\n")
            f.write("13 14 15 16 17 18 19 20 21 22 23 24 25\n")
            f.write("26 27 28 29 30 31 32 33 34 35 36 37 38\n")
            f.write("39 40 41 42 43 44 45 46 47 48 49 50 51\n")
            f.write("55 52\n")
            temp_path = f.name
        
        try:
            # Should detect as numeric format (no pipes in first non-empty line)
            records = load_dataset(temp_path)
            self.assertEqual(len(records), 1)
            # Should have parsed the numeric format correctly
            hand, auction, _ = records[0]
            self.assertIn("S:", hand)
            self.assertIn("H:", hand)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)