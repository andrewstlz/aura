"""
Test script for NLP interface improvements
"""

from NLP_interface import NaturalLanguageInterface

# Test cases
test_requests = [
    "please slim my face a little, smooth my face hard, extremely enlarge my eyes, add pink lipstick, and draw some eyeliner, thank you!"
]

def test_nlp_parsing(api_key: str = "YOUR_API_KEY_HERE"):
    """Test NLP parsing with various eyeliner requests"""
    
    nlp = NaturalLanguageInterface(api_key)
    
    print("="*70)
    print("NLP PARSING TEST - EYELINER DETECTION")
    print("="*70)
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}")
        print(f"{'='*70}")
        print(f'User: "{request}"')
        
        params = nlp.parse_request(request)
        params = nlp.validate_and_normalize(params)
        nlp.print_summary(params)
        
        # Check if eyeliner was detected
        makeup_config = params.get("makeup_config", {})
        eyeliner_thickness = makeup_config.get("eyeliner_thickness", 0)
        
        if eyeliner_thickness > 0:
            print(f"Eyeliner detected: thickness={eyeliner_thickness}")
        else:
            print("Eyeliner NOT detected (expected to be detected)")


if __name__ == "__main__":
    # Run actual test:
    test_nlp_parsing("YOUR_API_KEY_HERE")