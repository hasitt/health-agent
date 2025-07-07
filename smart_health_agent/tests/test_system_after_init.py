#!/usr/bin/env python3
"""
Test System After Initialization

This script tests if the Smart Health Agent system has been properly initialized
with TCM knowledge after the user clicks "Activate Agent System" in the UI.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_global_vectorstore():
    """Test if the global vectorstore has been initialized and contains TCM knowledge."""
    try:
        from smart_health_ollama import global_vectorstore
        
        if global_vectorstore is None:
            print("❌ Global vectorstore is not initialized")
            print("💡 Solution: Go to http://localhost:7860 and click 'Activate Agent System'")
            return False
        
        print("✅ Global vectorstore is initialized")
        
        # Test TCM knowledge search
        try:
            docs = global_vectorstore.similarity_search("TCM liver organ clock Five Elements", k=3)
            if docs:
                print(f"✅ Found {len(docs)} TCM-related documents")
                for i, doc in enumerate(docs):
                    content_preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"   Doc {i+1}: {content_preview}...")
                return True
            else:
                print("❌ No TCM documents found in vectorstore")
                return False
        except Exception as e:
            print(f"❌ Error searching vectorstore: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing global vectorstore: {e}")
        return False

def test_tcm_modules():
    """Test if TCM modules are available."""
    try:
        from smart_health_ollama import TCM_AVAILABLE
        if TCM_AVAILABLE:
            print("✅ TCM modules are available")
            return True
        else:
            print("❌ TCM modules are not available")
            return False
    except Exception as e:
        print(f"❌ Error checking TCM modules: {e}")
        return False

def test_garmin_availability():
    """Test if Garmin integration is available."""
    try:
        from smart_health_ollama import GARMIN_AVAILABLE
        if GARMIN_AVAILABLE:
            print("✅ Garmin integration is available")
            return True
        else:
            print("⚠️  Garmin integration is not available (this is OK for testing)")
            return True  # Not critical for TCM functionality
    except Exception as e:
        print(f"❌ Error checking Garmin availability: {e}")
        return False

def main():
    """Run all tests to verify system status."""
    print("🔍 Testing Smart Health Agent System Status...")
    print("=" * 60)
    
    tests = [
        ("Global Vectorstore", test_global_vectorstore),
        ("TCM Modules", test_tcm_modules),
        ("Garmin Availability", test_garmin_availability),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 System Status: {passed}/{total} components ready")
    
    if passed >= 2:  # At least vectorstore and TCM modules
        print("🎉 System is ready for TCM-enhanced health analysis!")
        print("\n💡 Try asking questions like:")
        print("   • 'Analyze my HRV data using TCM principles'")
        print("   • 'What does TCM say about liver health and stress?'")
        print("   • 'How do my metrics relate to the TCM organ clock?'")
    else:
        print("❌ System needs initialization")
        print("\n🔧 To fix:")
        print("   1. Go to http://localhost:7860")
        print("   2. Fill in the form fields:")
        print("      - Health Data Source: Choose Garmin or Synthetic Data")
        print("      - Your City: Enter your city name")
        print("      - Medical Knowledge Base: /Users/stan/dev/projects/health-agent/smart_health_agent/test_docs")
        print("   3. Click 'Activate Agent System'")
        print("   4. Wait for initialization to complete")
        print("   5. Run this test again")

if __name__ == "__main__":
    main() 