#!/usr/bin/env python3
"""
Test compatibility layer for old bridge_llm_bench.py API.

This provides a compatibility layer that maps old function calls to the new modular structure.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import the new modular structure
from bridge_llm_bench.parsers.bid_parser import parse_bid_from_response, _extract_bid_from_partial_response
from bridge_llm_bench.parsers.data_loader import load_dataset, _decode_hand, get_bid_from_id
from bridge_llm_bench.clients import get_client, OpenAIClient, AnthropicClient, GeminiClient
from bridge_llm_bench.metrics.evaluator import evaluate, build_prompt
from bridge_llm_bench.utils.file_utils import append_to_jsonl


class TestBridgeDataParsing(unittest.TestCase):
    """Test Bridge data parsing and conversion functions"""
    
    def test_bid_id_to_string_conversion(self):
        """Test conversion of numeric bid IDs to string representation"""
        # Test standard bid mappings
        self.assertEqual(get_bid_from_id(52), "Pass")
        self.assertEqual(get_bid_from_id(53), "X")
        self.assertEqual(get_bid_from_id(54), "XX")
        
        # Test suit bids (55 = 1C, 56 = 1D, etc.)
        self.assertEqual(get_bid_from_id(55), "1C")
        self.assertEqual(get_bid_from_id(56), "1D")
        self.assertEqual(get_bid_from_id(59), "1NT")
        self.assertEqual(get_bid_from_id(60), "2C")
        
        # Test edge cases
        self.assertEqual(get_bid_from_id(1), "1C")  # Alternative mapping
        self.assertEqual(get_bid_from_id(0), "Pass")  # Alternative mapping
        
        # Test unknown IDs
        self.assertTrue(get_bid_from_id(999).startswith("?"))
    
    def test_hand_decoding(self):
        """Test card ID to hand string conversion"""
        # Test with known card IDs
        # Spades: A=12, K=11, Q=10; Hearts: A=25, etc.
        cards = [12, 11, 10, 25, 24, 38, 37, 51, 50, 49, 48, 47, 46]  # AKQ of spades, AK of hearts, etc.
        hand_str = _decode_hand(cards)
        
        # Should contain suit separators and be properly formatted
        self.assertIn("S:", hand_str)
        self.assertIn("H:", hand_str)
        self.assertIn("D:", hand_str)
        self.assertIn("C:", hand_str)
    
    def test_text_data_loading(self):
        """Test loading Bridge data from text format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("S:AKQ H:987 D:AK C:5432 | 1H Pass | 2H\n")
            f.write("S:J987 H:AK D:Q987 C:43 | 1NT Pass | 2C\n")
            temp_path = f.name
        
        try:
            records = load_dataset(temp_path, n_records=2)
            self.assertEqual(len(records), 2)
            
            hand, auction, bid = records[0]
            self.assertIn("S:", hand)
            self.assertEqual(auction, "1H Pass")
            self.assertEqual(bid, "2H")
        finally:
            os.unlink(temp_path)
    
    def test_numeric_data_loading(self):
        """Test loading Bridge data from numeric format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write a simple numeric format: 13 cards for South, auction IDs, answer ID
            f.write("0 1 2 3 4 5 6 7 8 9 10 11 12\n")  # 13 cards for South
            f.write("13 14 15 16 17 18 19 20 21 22 23 24 25\n")  # North (ignored)
            f.write("26 27 28 29 30 31 32 33 34 35 36 37 38\n")  # East (ignored)  
            f.write("39 40 41 42 43 44 45 46 47 48 49 50 51\n")  # West (ignored)
            f.write("55 52\n")  # Auction: 1C Pass
            f.write("56\n")    # Answer: 1D
            temp_path = f.name
        
        try:
            records = load_dataset(temp_path, n_records=1)
            self.assertEqual(len(records), 1)
            
            hand, auction, bid = records[0]
            self.assertIn("S:", hand)  # Should have decoded hand
            self.assertIn("1C", auction)  # Should have converted bid IDs
            self.assertEqual(bid, "1D")  # Should have converted answer
        finally:
            os.unlink(temp_path)


class TestBidParsing(unittest.TestCase):
    """Test bid parsing from LLM responses"""
    
    def test_standard_bid_parsing(self):
        """Test parsing of standard Bridge bids"""
        test_cases = [
            ("1NT", "1NT"),
            ("2C", "2C"),
            ("7S", "7S"),
            ("Pass", "Pass"),
            ("P", "Pass"),
            ("X", "X"),
            ("XX", "XX"),
            ("Double", "X"),
            ("Redouble", "XX"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = parse_bid_from_response(input_text)
                self.assertEqual(result, expected)
    
    def test_malformed_response_parsing(self):
        """Test parsing of malformed LLM responses"""
        test_cases = [
            ("?42", "2C"),  # Should extract the number and assume clubs
            ("The bid is 1NT", "1NT"),  # Should extract from sentence
            ("I recommend passing", "Pass"),  # Should recognize pass
            ("1N", "1NT"),  # Should handle missing T
            ("3", "3C"),  # Should assume clubs for bare number
            ("", "?"),  # Empty string
            ("garbage text", "?"),  # No recognizable bid
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = parse_bid_from_response(input_text)
                if expected == "?":
                    self.assertEqual(result, "?")
                else:
                    self.assertEqual(result, expected)
    
    def test_partial_response_extraction(self):
        """Test extraction from partial/truncated responses"""
        test_cases = [
            ("1C with strong support", "1C"),
            ("Pass due to weak hand", "Pass"),
            ("Double for penalty", "X"),
            ("1N (truncated)", "1NT"),
            ("The answer is 2", "2C"),  # Should default to clubs
            ("Spades at the 1 level", "1S"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = _extract_bid_from_partial_response(input_text, input_text.upper())
                self.assertEqual(result, expected)


class TestLLMClients(unittest.TestCase):
    """Test LLM client implementations"""
    
    def test_client_factory(self):
        """Test LLM client factory function"""
        # Test valid model prefixes
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('bridge_llm_bench.clients.openai_client.openai'):
                client = get_client("gpt-4", 0.0)
                self.assertIsInstance(client, OpenAIClient)
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('anthropic.Anthropic'):
                client = get_client("claude-3-opus", 0.0)
                self.assertIsInstance(client, AnthropicClient)
        
        # Test invalid model prefix
        with self.assertRaises(ValueError):
            get_client("invalid-model", 0.0)
    
    @patch('bridge_llm_bench.clients.gemini_client.genai')
    def test_gemini_client_initialization(self, mock_genai):
        """Test Gemini client initialization and configuration"""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
            client = GeminiClient("gemini-1.5-pro", 0.1)
            
            # Verify safety settings were configured
            self.assertEqual(len(client.safety_settings), 4)
            self.assertTrue(all(setting["threshold"] == "BLOCK_NONE" 
                              for setting in client.safety_settings))
    
    @patch('bridge_llm_bench.clients.gemini_client.genai')
    def test_gemini_client_response_handling(self, mock_genai):
        """Test Gemini client response handling and retries"""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
            client = GeminiClient("gemini-1.5-pro", 0.1)
            
            # Mock successful response with proper structure
            mock_candidate = Mock()
            mock_candidate.finish_reason = 1  # STOP
            
            mock_response = Mock()
            mock_response.text = "1NT"
            mock_response.candidates = [mock_candidate]
            mock_response.usage_metadata = Mock()
            mock_response.usage_metadata.prompt_token_count = 10
            mock_response.usage_metadata.candidates_token_count = 5
            
            client.model_ref.generate_content.return_value = mock_response
            
            text, metadata = client.get_completion("test prompt")
            
            self.assertEqual(text, "1NT")
            self.assertEqual(metadata["prompt_tokens"], 10)
            self.assertEqual(metadata["completion_tokens"], 5)
    
    @patch('bridge_llm_bench.clients.gemini_client.genai')
    def test_gemini_client_empty_response_handling(self, mock_genai):
        """Test Gemini client handling of empty responses"""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
            client = GeminiClient("gemini-1.5-pro", 0.1)
            
            # Mock empty response
            mock_response = Mock()
            mock_response.text = ""
            mock_response.usage_metadata = None
            
            client.model_ref.generate_content.return_value = mock_response
            
            # Should return fallback
            text, metadata = client.get_completion("test prompt")
            
            self.assertEqual(text, "Pass")
            self.assertEqual(metadata["prompt_tokens"], 0)


class TestBenchmarkExecution(unittest.TestCase):
    """Test benchmark execution and metrics calculation"""
    
    def setUp(self):
        """Set up test data for benchmark tests"""
        self.test_records = [
            ("S:AKQ H:987 D:AK C:5432", "1H Pass", "2H"),
            ("S:J987 H:AK D:Q987 C:43", "1NT Pass", "2C"),
            ("S:A987 H:K987 D:A987 C:A", "", "1C"),
        ]
    
    def test_prompt_building(self):
        """Test Bridge prompt construction"""
        hand = "S:AKQ H:987 D:AK C:5432"
        auction = "1H Pass"
        convention = "SAYC"
        
        prompt = build_prompt(hand, auction, convention)
        
        self.assertIn("Standard American Yellow Card", prompt)
        self.assertIn(hand, prompt)
        self.assertIn(auction, prompt)
        self.assertIn("Your call?", prompt)
    
    @patch('bridge_llm_bench.metrics.evaluator.get_client')
    def test_evaluation_function(self, mock_get_client):
        """Test evaluation of model performance"""
        # Mock client
        mock_client = Mock()
        mock_client.get_completion.return_value = ("2H", {"prompt_tokens": 50, "completion_tokens": 5})
        mock_get_client.return_value = mock_client
        
        summary, confusion = evaluate(
            self.test_records[:1], 
            "test-model", 
            "SAYC"
        )
        
        # Verify summary statistics
        self.assertEqual(summary["model"], "test-model")
        self.assertEqual(summary["convention"], "SAYC")
        self.assertEqual(summary["n_records"], 1)
        self.assertEqual(summary["accuracy"], 1.0)  # Perfect match
        
        # Verify confusion matrix
        self.assertEqual(confusion["2H"]["2H"], 1)
    
    @patch('bridge_llm_bench.metrics.evaluator.get_client')
    def test_evaluation_with_errors(self, mock_get_client):
        """Test evaluation handling of API errors"""
        # Mock client that throws errors
        mock_client = Mock()
        mock_client.get_completion.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client
        
        summary, confusion = evaluate(
            self.test_records[:1], 
            "test-model", 
            "SAYC"
        )
        
        # Should handle errors gracefully
        self.assertEqual(summary["accuracy"], 0.0)  # No correct predictions due to errors
        self.assertEqual(summary["n_records"], 1)


class TestDataExport(unittest.TestCase):
    """Test data export functionality"""
    
    def test_jsonl_export(self):
        """Test JSONL file export"""
        test_data = {"test": "data", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            append_to_jsonl(temp_path, test_data)
            
            # Read back and verify
            with open(temp_path, 'r') as f:
                line = f.readline().strip()
                loaded_data = json.loads(line)
                self.assertEqual(loaded_data, test_data)
        finally:
            temp_path.unlink()


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_dataset_loading_and_processing(self):
        """Test complete dataset loading and processing workflow"""
        # Create a small test dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("S:AKQ H:987 D:AK C:5432 | 1H Pass | 2H\n")
            f.write("S:J987 H:AK D:Q987 C:43 | 1NT Pass | 2C\n")
            temp_path = f.name
        
        try:
            # Load dataset
            records = load_dataset(temp_path, n_records=2)
            self.assertEqual(len(records), 2)
            
            # Verify data integrity
            for hand, auction, bid in records:
                self.assertIn("S:", hand)
                self.assertTrue(len(auction) >= 0)  # Can be empty
                self.assertTrue(len(bid) > 0)
                
        finally:
            os.unlink(temp_path)
    
    @patch('bridge_llm_bench.metrics.evaluator.get_client')
    def test_end_to_end_benchmark(self, mock_get_client):
        """Test end-to-end benchmark execution"""
        # Mock client responses
        mock_client = Mock()
        responses = [
            ("2H", {"prompt_tokens": 50, "completion_tokens": 5}),
            ("2C", {"prompt_tokens": 45, "completion_tokens": 4}),
        ]
        mock_client.get_completion.side_effect = responses
        mock_get_client.return_value = mock_client
        
        records = [
            ("S:AKQ H:987 D:AK C:5432", "1H Pass", "2H"),
            ("S:J987 H:AK D:Q987 C:43", "1NT Pass", "2C"),
        ]
        
        # Run evaluation
        summary, confusion = evaluate(records, "test-model", "SAYC")
        
        # Verify results
        self.assertEqual(summary["accuracy"], 1.0)  # Perfect predictions
        self.assertEqual(summary["n_records"], 2)
        self.assertEqual(summary["prompt_tokens"], 95)
        self.assertEqual(summary["completion_tokens"], 9)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_invalid_dataset_handling(self):
        """Test handling of invalid or corrupted datasets"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid data format\n")
            f.write("no pipes or proper structure\n")
            temp_path = f.name
        
        try:
            records = load_dataset(temp_path)
            # Should return empty list or handle gracefully
            self.assertIsInstance(records, list)
        finally:
            os.unlink(temp_path)
    
    def test_missing_api_keys(self):
        """Test handling of missing API keys"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                GeminiClient("gemini-1.5-pro", 0.0)
    
    def test_edge_case_bids(self):
        """Test parsing of edge case bid formats"""
        edge_cases = [
            "1 NT",  # Space in NT
            "1n",    # Lowercase
            "PASS ",  # Trailing space
            " X",     # Leading space
            "2♣",     # Unicode suit symbol (if supported)
        ]
        
        for case in edge_cases:
            result = parse_bid_from_response(case)
            # Should not crash and return something reasonable
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)


if __name__ == '__main__':
    # Configure test runner for detailed output
    unittest.main(verbosity=2, buffer=True)