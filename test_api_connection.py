#!/usr/bin/env python3
"""Test API connection"""
import sys
import os

sys.path.insert(0, '/usr1/home/s125mdg43_10/AutoFusion_v3/src')

print('Step 1: Import start')
from utils.llm_backend import UnifiedLLMBackend
print('Step 2: LLM Backend imported')

key = os.environ.get('ALIYUN_API_KEY')
print(f'Step 3: API Key present: {bool(key)}')
if key:
    print(f'Key prefix: {key[:10]}...')

llm = UnifiedLLMBackend()
print('Step 4: LLM initialized')

result = llm.generate('Generate a simple Python function that adds two numbers.')
print(f'Step 5: Generation complete, code length: {len(result.code)}')
print('SUCCESS!')
