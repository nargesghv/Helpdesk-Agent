"""
Test client for Helpdesk Agent API
Demonstrates how to call the backend endpoints
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8004"

def test_health():
    """Test health check endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_single_ticket():
    """Test single ticket processing"""
    print("🎫 Testing single ticket processing...")
    
    ticket = {
        "issue": "VPN connection timeout",
        "description": "VPN disconnects after 5 minutes of usage. Tried reconnecting multiple times.",
        "category": "Network"
    }
    
    response = requests.post(
        f"{BASE_URL}/process/single",
        json=ticket
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\n📋 Issue: {result['issue']}")
        print(f"\n💡 Direction:")
        print(result['direction_bullets'])
        print(f"\n📚 Evidence ({len(result['evidence'])} tickets):")
        for e in result['evidence'][:3]:  # Show first 3
            print(f"  - {e['ticket_id']}: {e['Issue']} (Resolved: {e['Resolved']})")
    else:
        print(f"Error: {response.text}")
    print()

def test_batch_tickets():
    """Test batch ticket processing"""
    print("📦 Testing batch ticket processing...")
    
    batch = {
        "tickets": [
            {
                "issue": "Email not syncing",
                "description": "Outlook emails not syncing on mobile device",
                "category": "Email"
            },
            {
                "issue": "Printer offline",
                "description": "Cannot connect to network printer",
                "category": "Hardware"
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/process/batch",
        json=batch
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"\n✅ Processed {len(results)} tickets")
        for r in results:
            print(f"\n  Ticket {r['ticket_id']}: {r['issue']}")
            print(f"  Evidence: {len(r['evidence'])} similar tickets")
    else:
        print(f"Error: {response.text}")
    print()

def test_stats():
    """Test stats endpoint"""
    print("📊 Testing stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Helpdesk Agent API Test Client")
    print("=" * 60)
    print()
    
    try:
        # Test health
        test_health()
        
        # Test stats
        test_stats()
        
        # Test single ticket
        test_single_ticket()
        
        # Test batch
        test_batch_tickets()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API at http://localhost:8004")
        print("   Make sure the backend is running:")
        print("   python backend.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")

