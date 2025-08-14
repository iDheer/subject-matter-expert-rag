import requests
import json

BASE_URL = "http://localhost:8000"

def test_status():
    """Test the status endpoint"""
    print("🔍 Testing /status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return False

def test_query(question: str, use_memory: bool = True):
    """Test the query endpoint"""
    print(f"🤖 Testing query: '{question}'")
    
    payload = {
        "question": question,
        "use_memory": use_memory,
        "stream": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Sources: {len(result['sources'])} found")
            print(f"Memory: {result['memory_status']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False
    finally:
        print()

def test_memory_operations():
    """Test memory operations"""
    print("🧠 Testing memory operations...")
    
    try:
        # Get current status
        response = requests.get(f"{BASE_URL}/memory/status")
        print(f"Initial memory status: {response.json()}")
        
        # Toggle memory
        response = requests.post(f"{BASE_URL}/memory/toggle")
        print(f"Toggle result: {response.json()}")
        
        # Clear memory
        response = requests.post(f"{BASE_URL}/memory/clear")
        print(f"Clear result: {response.json()}")
        print()
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        print("🧪 Testing SME API...")
        
        # Test basic functionality
        if not test_status():
            print("❌ Status test failed, stopping tests")
            exit(1)
        
        # Test queries
        success = True
        success &= test_query("What is the main topic of the documents?")
        success &= test_query("Can you elaborate on that?")  # This should use conversation context
        
        # Test memory operations
        success &= test_memory_operations()
        
        if success:
            print("✅ All API tests passed!")
        else:
            print("⚠️ Some tests failed")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on localhost:8000")
        print("   Run: python api_server.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")