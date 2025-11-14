#!/usr/bin/env python3
"""Test script to check vLLM API compatibility"""
import requests
import json

# Test different parameter combinations
test_cases = [
    {
        "name": "with_top_k_and_min_p",
        "params": {
            "model": "models/Qwen3-1.7B",
            "prompt": "Hello, who are you?",
            "max_tokens": 100,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "n": 1,
            "logprobs": 1,
        }
    },
    {
        "name": "without_top_k_and_min_p",
        "params": {
            "model": "models/Qwen3-1.7B",
            "prompt": "Hello, who are you?",
            "max_tokens": 100,
            "temperature": 0.6,
            "top_p": 0.95,
            "n": 1,
            "logprobs": 1,
        }
    },
    {
        "name": "with_top_k_only",
        "params": {
            "model": "models/Qwen3-1.7B",
            "prompt": "Hello, who are you?",
            "max_tokens": 100,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "n": 1,
            "logprobs": 1,
        }
    },
]

url = "http://127.0.0.1:8201/v1/completions"

for test in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing: {test['name']}")
    print(f"{'='*60}")
    print("Request params:")
    print(json.dumps(test['params'], indent=2))
    
    try:
        response = requests.post(url, json=test['params'], timeout=30)
        print(f"\nStatus Code: {response.status_code}")
        if response.status_code == 200:
            print("✓ SUCCESS")
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                print(f"Response text: {result['choices'][0]['text'][:100]}...")
        else:
            print("✗ FAILED")
            print(f"Error response: {response.text[:500]}")
    except Exception as e:
        print(f"✗ Exception: {e}")

print(f"\n{'='*60}")

