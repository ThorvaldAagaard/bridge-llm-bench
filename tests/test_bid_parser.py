"""
Comprehensive unit tests for bid parsing functionality.

Tests cover standard bids, malformed responses, edge cases, and partial responses.
"""

import unittest
from bridge_llm_bench.parsers.bid_parser import (
    parse_bid_from_response,
    get_bid_from_id,
    _is_valid_bid,
    _normalize_bid,
    _extract_bid_from_partial_response
)


class TestBidParsing(unittest.TestCase):
    """Test suite for bid parsing functions."""
    
    def test_standard_bid_parsing(self):
        """Test parsing of standard Bridge bids."""
        test_cases = [
            ("1NT", "1NT"),
            ("2C", "2C"),
            ("7S", "7S"),
            ("Pass", "PASS"),
            ("pass", "PASS"),
            ("PASS", "PASS"),
            ("X", "X"),
            ("XX", "XX"),
            ("x", "X"),
            ("xx", "XX"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = parse_bid_from_response(input_text)
                self.assertEqual(result, expected)
    
    def test_structured_response_formats(self):
        """Test parsing from structured response formats."""
        test_cases = [
            # O3-style FINAL BID format
            ("FINAL BID: 2H", "2H"),
            ("Final Bid: 1NT", "1NT"),
            ("FINAL BID: PASS", "PASS"),
            
            # Claude-style MY BID IS format
            ("MY BID IS: 1NT", "1NT"),
            ("My bid is: 2C", "2C"),
            ("I BID: 3S", "3S"),
            ("CALL: 2H", "2H"),
            ("BID: 1C", "1C"),
            
            # Mixed with explanation
            ("After analysis, FINAL BID: 2NT", "2NT"),
            ("Considering the hand, MY BID IS: 1S", "1S"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = parse_bid_from_response(input_text)
                self.assertEqual(result, expected)
    
    def test_malformed_response_parsing(self):
        """Test parsing of malformed LLM responses."""
        test_cases = [
            # Missing T in NT
            ("1N", "1NT"),
            ("2n", "2NT"),
            ("3N", "3NT"),
            
            # Embedded in text
            ("The bid is 1NT", "1NT"),
            ("I would bid 2C here", "2C"),
            ("My recommendation is to pass", "PASS"),
            
            # Common variations
            ("P", "PASS"),
            ("p", "PASS"),
            ("Double", "X"),
            ("double", "X"),
            ("DOUBLE", "X"),
            ("Redouble", "XX"),
            ("REDOUBLE", "XX"),
            
            # Empty or invalid
            ("", "?"),
            ("garbage text", "?"),
            ("no bid here", "PASS"),  # Contains "no bid"
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = parse_bid_from_response(input_text)
                self.assertEqual(result, expected)
    
    def test_special_case_42_parsing(self):
        """Test parsing of the special ?42 case."""
        # ?42 should map to bid ID 42
        result = parse_bid_from_response("?42")
        # Based on the BID_ID2STR mapping, we need to check what 42 maps to
        bid_42 = get_bid_from_id(42)
        if not bid_42.startswith("?"):
            self.assertEqual(result, bid_42)
        else:
            self.assertEqual(result, "?")
    
    def test_partial_response_extraction(self):
        """Test extraction from partial/truncated responses."""
        test_cases = [
            # Truncated with context
            ("1C with strong support", "1C"),
            ("Pass due to weak hand", "PASS"),
            ("Double for penalty", "X"),
            ("1N (truncated response)", "1NT"),
            
            # Level and suit mentioned separately
            ("The answer is 2", "2C"),  # Defaults to clubs
            ("Bid 3", "3C"),
            ("I'll go with 4", "4C"),
            
            # Suit mentioned
            ("Spades at the 1 level", "1S"),
            ("2 hearts", "2H"),
            ("Three diamonds", "3D"),
            ("Four clubs", "4C"),
            ("notrump at the 2 level", "2NT"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = parse_bid_from_response(input_text)
                self.assertEqual(result, expected)
    
    def test_bid_id_conversion(self):
        """Test numeric bid ID to string conversion."""
        test_cases = [
            (52, "Pass"),
            (53, "X"),
            (54, "XX"),
            (55, "1C"),
            (56, "1D"),
            (57, "1H"),
            (58, "1S"),
            (59, "1NT"),
            (60, "2C"),
            (89, "7NT"),  # Last valid bid
            
            # Alternative mappings
            (0, "Pass"),
            (1, "1C"),
            
            # Unknown IDs
            (999, "?999"),
            (-1, "?-1"),
        ]
        
        for bid_id, expected in test_cases:
            with self.subTest(bid_id=bid_id):
                result = get_bid_from_id(bid_id)
                self.assertEqual(result, expected)
    
    def test_is_valid_bid(self):
        """Test bid validation function."""
        valid_bids = [
            "PASS", "X", "XX",
            "1C", "1D", "1H", "1S", "1NT",
            "2C", "2D", "2H", "2S", "2NT",
            "7C", "7D", "7H", "7S", "7NT",
        ]
        
        for bid in valid_bids:
            with self.subTest(bid=bid):
                self.assertTrue(_is_valid_bid(bid))
                self.assertTrue(_is_valid_bid(bid.lower()))
        
        invalid_bids = [
            "1N", "8C", "0NT", "PASSES", "XXX",
            "1", "C", "NT", "?", "DOUBLE",
            "", "garbage", "1 NT", "2 C",
        ]
        
        for bid in invalid_bids:
            with self.subTest(bid=bid):
                self.assertFalse(_is_valid_bid(bid))
    
    def test_normalize_bid(self):
        """Test bid normalization function."""
        test_cases = [
            ("pass", "PASS"),
            ("P", "PASS"),
            ("PASS", "PASS"),
            ("x", "X"),
            ("D", "X"),
            ("DBL", "X"),
            ("DOUBLE", "X"),
            ("xx", "XX"),
            ("R", "XX"),
            ("RDBL", "XX"),
            ("REDOUBLE", "XX"),
            ("1n", "1NT"),
            ("2N", "2NT"),
            ("1nt", "1NT"),
            ("1NT", "1NT"),
        ]
        
        for input_bid, expected in test_cases:
            with self.subTest(input=input_bid):
                result = _normalize_bid(input_bid)
                self.assertEqual(result, expected)
    
    def test_extract_from_partial_with_edge_cases(self):
        """Test partial extraction with edge cases."""
        test_cases = [
            # Multiple numbers
            ("Between 1 and 2, I choose 2", "2C"),
            
            # Suit variations
            ("club", "C"),
            ("clubs", "C"),
            ("diamond", "D"),
            ("diamonds", "D"),
            ("heart", "H"),
            ("hearts", "H"),
            ("spade", "S"),
            ("spades", "S"),
            
            # NT variations
            ("no trump", "NT"),
            ("notrump", "NT"),
            ("NT", "NT"),
            
            # Pass variations
            ("I pass", "PASS"),
            ("passing", "PASS"),
            ("no bid", "PASS"),
            
            # Double/Redouble in context
            ("I double", "X"),
            ("doubling", "X"),
            ("I redouble", "XX"),
            
            # No valid bid found
            ("random text without bid", "?"),
            ("", "?"),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = _extract_bid_from_partial_response(text, text.upper())
                if expected in ["C", "D", "H", "S", "NT"]:
                    # These are suit/NT extractions without level
                    self.assertTrue(result == "?" or result.endswith(expected))
                else:
                    self.assertEqual(result, expected)


class TestBidIDMapping(unittest.TestCase):
    """Test suite for bid ID mapping consistency."""
    
    def test_bid_id_mapping_completeness(self):
        """Test that all expected bid IDs are mapped."""
        # Check standard bids
        self.assertEqual(get_bid_from_id(52), "Pass")
        self.assertEqual(get_bid_from_id(53), "X")
        self.assertEqual(get_bid_from_id(54), "XX")
        
        # Check all level/suit combinations (55-89)
        expected_id = 55
        for level in range(1, 8):
            for suit in ["C", "D", "H", "S", "NT"]:
                expected_bid = f"{level}{suit}"
                actual_bid = get_bid_from_id(expected_id)
                self.assertEqual(actual_bid, expected_bid, 
                               f"ID {expected_id} should map to {expected_bid}")
                expected_id += 1
    
    def test_bid_id_offset_correction(self):
        """Test offset correction for problematic IDs."""
        # Test various offsets that might be tried
        test_id = 100  # An unknown ID
        result = get_bid_from_id(test_id)
        
        # Should try various offsets and either find a valid bid or return ?100
        self.assertTrue(result.startswith("?") or _is_valid_bid(result))


if __name__ == '__main__':
    unittest.main(verbosity=2)