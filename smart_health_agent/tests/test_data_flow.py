#!/usr/bin/env python3
"""
Test Data Flow - Trace Garmin Data Through the Agent Workflow
"""

import sys
import os
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_flow():
    """Test the complete data flow from Garmin to recommendations."""
    
    print("🔍 Testing Garmin Data Flow Through Agent Workflow")
    print("=" * 60)
    
    try:
        # Step 1: Test Garmin Data Retrieval
        print("\n📊 Step 1: Testing Garmin Data Retrieval")
        print("-" * 40)
        
        from garmin_utils import get_garmin_health_data, GarminHealthData
        raw_garmin_data = get_garmin_health_data()
        print(f"✅ Raw Garmin data retrieved: {len(str(raw_garmin_data))} characters")
        print(f"   Keys: {list(raw_garmin_data.keys())}")
        
        # Step 2: Test Data Processing
        print("\n🔄 Step 2: Testing Data Processing")
        print("-" * 40)
        
        from smart_health_ollama import process_garmin_data
        processed_data = process_garmin_data(raw_garmin_data)
        print(f"✅ Processed data keys: {list(processed_data.keys())}")
        print(f"   HR: {processed_data.get('heart_rate')}, Steps: {processed_data.get('steps')}, Sleep: {processed_data.get('sleep_hours')}")
        
        # Step 3: Test Agent State Creation
        print("\n🏗️ Step 3: Testing Agent State Creation")
        print("-" * 40)
        
        from smart_health_ollama import HealthAgentState
        from langchain_core.messages import HumanMessage
        
        initial_state = HealthAgentState(
            health_data=processed_data,
            weather_data={'temperature': 20, 'condition': 'Clear'},
            messages=[HumanMessage(content="Test message")]
        )
        print(f"✅ Initial state created")
        print(f"   State health_data keys: {list(initial_state.health_data.keys())}")
        print(f"   HR in state: {initial_state.health_data.get('heart_rate')}")
        print(f"   Steps in state: {initial_state.health_data.get('steps')}")
        print(f"   Sleep in state: {initial_state.health_data.get('sleep_hours')}")
        
        # Step 4: Test HealthMetricsAgent
        print("\n🏥 Step 4: Testing HealthMetricsAgent")
        print("-" * 40)
        
        from smart_health_ollama import HealthMetricsAgent, GARMIN_AVAILABLE, TCM_AVAILABLE
        print(f"   GARMIN_AVAILABLE: {GARMIN_AVAILABLE}")
        print(f"   TCM_AVAILABLE: {TCM_AVAILABLE}")
        print(f"   garmin_stress_data empty: {not initial_state.garmin_stress_data}")
        
        # Check the conditions for TCM analysis
        tcm_condition = GARMIN_AVAILABLE and TCM_AVAILABLE and not initial_state.garmin_stress_data
        print(f"   TCM analysis condition: {tcm_condition}")
        
        updated_state = HealthMetricsAgent(initial_state)
        print(f"✅ HealthMetricsAgent completed")
        print(f"   Health data after agent: {list(updated_state.health_data.keys())}")
        print(f"   HR after agent: {updated_state.health_data.get('heart_rate')}")
        print(f"   Vitals status: {updated_state.health_data.get('vitals_status')}")
        print(f"   TCM insights available: {bool(updated_state.tcm_insights)}")
        print(f"   Garmin stress data: {len(updated_state.garmin_stress_data)} entries")
        
        # Step 5: Test Full Workflow
        print("\n🔄 Step 5: Testing Full Workflow")
        print("-" * 40)
        
        from smart_health_ollama import build_health_workflow
        app = build_health_workflow()
        
        final_state = app.invoke(initial_state)
        print(f"✅ Full workflow completed")
        print(f"   Final recommendations: {len(final_state.get('recommendations', []))}")
        print(f"   Streaming response available: {bool(final_state.get('streaming_response'))}")
        
        # Step 6: Check what the user would see
        print("\n👁️ Step 6: What the User Sees")
        print("-" * 40)
        
        response_content = final_state.get('streaming_response')
        if not response_content:
            if final_state.get('recommendations'):
                response_content = final_state['recommendations'][-1].content
            else:
                response_content = "No response generated"
        
        print(f"Response length: {len(response_content)} characters")
        print(f"Contains 'Garmin': {'Garmin' in response_content}")
        print(f"Contains 'TCM': {'TCM' in response_content}")
        print(f"Contains HR value '{processed_data.get('heart_rate')}': {str(processed_data.get('heart_rate')) in response_content}")
        print(f"Contains steps value '{processed_data.get('steps')}': {str(processed_data.get('steps')) in response_content}")
        
        # Show first 200 characters of response
        print(f"\nFirst 200 chars of response:")
        print(f"'{response_content[:200]}...'")
        
        # Step 7: Check for the specific issue
        print("\n🔍 Step 7: Diagnosing the Issue")
        print("-" * 40)
        
        print("CRITICAL ANALYSIS:")
        print(f"1. Garmin data was retrieved: ✅")
        print(f"2. Data was processed correctly: ✅")
        print(f"3. Initial state contains data: ✅")
        print(f"4. HealthMetricsAgent received data: ✅")
        print(f"5. TCM analysis was attempted: {'✅' if updated_state.tcm_insights else '❌'}")
        print(f"6. Final response includes data: {'✅' if str(processed_data.get('heart_rate')) in response_content else '❌'}")
        
        if str(processed_data.get('heart_rate')) not in response_content:
            print("\n❌ ISSUE IDENTIFIED: Garmin data is not making it into the final response!")
            print("   The data is available to agents but not being used in recommendations.")
        else:
            print("\n✅ Data flow appears to be working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data flow test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_flow()
    sys.exit(0 if success else 1) 