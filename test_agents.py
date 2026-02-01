import unittest
import os
import pandas as pd
from app import run_agent_1, analyze_data # Aapki main file se functions import karein

class TestTeslaAI(unittest.TestCase):

    # TEST 1: Check if CSV file exists (Data Integrity Test)
    def test_csv_existence(self):
        self.assertTrue(os.path.exists("tesla_data.csv"), "Error: tesla_data.csv file missing!")

    # TEST 2: Check if Agent 1 produces clusters (Logic Test)
    def test_agent1_output(self):
        summary = run_agent_1()
        self.assertIn("3 clusters", summary, "Agent 1 should find 3 clusters.")
        self.assertIsInstance(summary, str, "Agent 1 output should be a string.")

    # TEST 3: Check if Gemini API is working (Integration Test)
    def test_ai_response(self):
        test_prompt = "Say hello"
        response = analyze_data(test_prompt)
        # Agar block ho jaye ya error aaye to ye test fail hoga
        self.assertNotEqual(response, "AI did not give an answer.", "Gemini API is blocked or not responding!")
        self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main()