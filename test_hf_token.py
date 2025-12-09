#!/usr/bin/env python3
"""
Test script to check if HF_TOKEN is valid and if it's hitting rate limits.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
import time

def get_hf_token():
    """Load HF token from environment or .env file."""
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path if env_path else None)
    tok = os.getenv("HF_TOKEN")
    if not tok:
        print("[ERROR] HF_TOKEN not found in environment")
        print("   Set it with: export HF_TOKEN=your_token")
        print("   Or add it to a .env file: HF_TOKEN=your_token")
        sys.exit(1)
    return tok

def load_config_models():
    """Load models from attack_llm_config.json."""
    # Try to find config file in multiple locations
    config_paths = [
        Path("config/attack_llm_config.json"),
        Path("MLOps-Project/config/attack_llm_config.json"),
        Path(__file__).parent.parent / "config" / "attack_llm_config.json",
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break
    
    if not config_path:
        print("[WARN] Could not find attack_llm_config.json, using default test models")
        return [
            {"model_id": "mistralai/Mistral-7B-Instruct-v0.2", "provider": None},
            {"model_id": "google/flan-t5-base", "provider": None},
        ]
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        models = config.get("models", [])
        return [
            {
                "model_id": m.get("model_id"),
                "provider": m.get("provider"),
                "name": m.get("name", "unknown")
            }
            for m in models if m.get("model_id")
        ]
    except Exception as e:
        print(f"[WARN] Error loading config: {e}, using default test models")
        return [
            {"model_id": "mistralai/Mistral-7B-Instruct-v0.2", "provider": None},
            {"model_id": "google/flan-t5-base", "provider": None},
        ]

def test_hf_token():
    """Test HF token by making a simple API call."""
    print("=" * 60)
    print("Testing HF_TOKEN...")
    print("=" * 60)
    
    token = get_hf_token()
    print(f"[OK] Token found: {token[:10]}...{token[-4:]}")
    print()
    
    # Load models from config file
    print("Loading models from attack_llm_config.json...")
    test_models = load_config_models()
    print(f"[OK] Found {len(test_models)} model(s) to test")
    print()
    
    test_prompt = "Say 'Hello, world!' in one sentence."
    
    last_error = None
    for model_config in test_models:
        model_id = model_config["model_id"]
        provider = model_config.get("provider")
        model_name = model_config.get("name", model_id)
        
        print(f"Testing with model: {model_name} ({model_id})")
        if provider:
            print(f"  Provider: {provider}")
        print(f"Test prompt: {test_prompt}")
        print()
        
        try:
            print("Creating InferenceClient...")
            client_kwargs = {
                "model": model_id,
                "token": token,
                "timeout": 30
            }
            if provider:
                client_kwargs["provider"] = provider
            
            client = InferenceClient(**client_kwargs)
            print("[OK] Client created successfully")
            print()
            
            print("Making inference request...")
            start_time = time.time()
            
            # Try chat_completion first (for instruction models), fallback to text_generation
            try:
                response = client.chat_completion(
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=50,
                    temperature=0.7
                )
                # Extract text from chat response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    response_text = response.choices[0].message.content
                else:
                    response_text = str(response)
            except Exception as chat_error:
                # Fallback to text_generation if chat doesn't work
                if "not supported" in str(chat_error).lower() or "conversational" in str(chat_error).lower():
                    response = client.text_generation(
                        prompt=test_prompt,
                        max_new_tokens=50,
                        temperature=0.7
                    )
                    response_text = response
                else:
                    raise chat_error
            
            elapsed = time.time() - start_time
            print(f"[OK] Request successful! (took {elapsed:.2f}s)")
            print(f"  Response: {str(response_text)[:100]}...")
            print()
            print("[SUCCESS] HF_TOKEN is working correctly - no rate limit issues detected!")
            return True
            
        except Exception as e:
            error_msg = str(e)
            last_error = e
            # If it's a 404 or task not supported, try the next model
            if "404" in error_msg or "not found" in error_msg.lower() or "doesn't support" in error_msg.lower():
                print(f"[INFO] Model {model_name} not available or incompatible, trying next model...")
                print()
                continue
            # For rate limit or auth errors, check them immediately
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                print(f"[RATE LIMIT] Detected with model {model_name}!")
                break
            if "401" in error_msg or "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
                print(f"[AUTH ERROR] Detected with model {model_name}!")
                break
            # For other errors, continue trying
            print(f"[INFO] Error with {model_name}: {error_msg[:80]}...")
            print("   Trying next model...")
            print()
            continue
    
    # If we get here, all models failed or we hit a non-404 error
    if last_error:
        error_msg = str(last_error)
        print(f"[ERROR] All test models failed. Last error: {error_msg}")
        print()
        
        # Check for specific error types
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            print("[RATE LIMIT] DETECTED!")
            print("   Your HF_TOKEN has hit the rate limit.")
            print("   You may need to:")
            print("   - Wait for the rate limit to reset")
            print("   - Upgrade your HuggingFace plan")
            print("   - Use a different token")
            return False
        elif "401" in error_msg or "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
            print("[AUTH ERROR] DETECTED!")
            print("   Your HF_TOKEN is invalid or expired.")
            print("   Please check your token and update it.")
            return False
        elif "403" in error_msg or "forbidden" in error_msg.lower():
            print("[PERMISSION ERROR] DETECTED!")
            print("   Your HF_TOKEN doesn't have permission to access this model.")
            print("   You may need to:")
            print("   - Accept the model's terms of use on HuggingFace")
            print("   - Request access to the model")
            return False
        elif "timeout" in error_msg.lower():
            print("[TIMEOUT ERROR] DETECTED!")
            print("   The request timed out. This might be a temporary issue.")
            print("   Try again in a few moments.")
            return False
        elif "404" in error_msg or "not found" in error_msg.lower() or "doesn't support" in error_msg.lower():
            print("[MODEL COMPATIBILITY] ISSUE DETECTED!")
            print("   The test models were not available or don't support the requested task.")
            print("   This might mean:")
            print("   - The models require special access")
            print("   - The models don't support text-generation")
            print("   - The models are not available on the inference API")
            print()
            print("   IMPORTANT: The token itself appears to be VALID (no auth errors).")
            print("   IMPORTANT: NO RATE LIMIT ERRORS detected (no 429 errors).")
            print("   The token should work fine with models that support your use case.")
            return True  # Token is valid, just couldn't test with these models
        else:
            print("[UNKNOWN ERROR] DETECTED!")
            print("   The error doesn't match common patterns.")
            print("   Full error details above.")
            print()
            print("   However, since we didn't see rate limit (429) or auth (401) errors,")
            print("   the token appears to be valid.")
            return False
    else:
        print("[ERROR] No models were tested or all models returned 404.")
        print("   Could not determine token status.")
        return False

if __name__ == "__main__":
    success = test_hf_token()
    sys.exit(0 if success else 1)

