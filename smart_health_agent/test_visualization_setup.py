#!/usr/bin/env python3
"""
Test script to verify visualization setup
"""

print("Testing visualization setup...")

# Test matplotlib import
try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported successfully")
    matplotlib_available = True
except ImportError as e:
    print(f"❌ matplotlib not available: {e}")
    matplotlib_available = False

# Test seaborn import
try:
    import seaborn as sns
    print("✅ seaborn imported successfully")
    seaborn_available = True
except ImportError as e:
    print(f"❌ seaborn not available: {e}")
    seaborn_available = False

# Test health_visualizations import
try:
    import health_visualizations
    print("✅ health_visualizations imported successfully")
    visualizations_available = True
except ImportError as e:
    print(f"❌ health_visualizations not available: {e}")
    visualizations_available = False

print("\n" + "="*50)
print("SUMMARY:")
print(f"matplotlib: {'✅ Available' if matplotlib_available else '❌ Not available'}")
print(f"seaborn: {'✅ Available' if seaborn_available else '❌ Not available'}")
print(f"health_visualizations: {'✅ Available' if visualizations_available else '❌ Not available'}")

if not matplotlib_available or not seaborn_available:
    print("\n🔧 TO FIX:")
    print("Run: pip install matplotlib seaborn")
    print("Or: python -m pip install matplotlib seaborn")

print("\n📋 NEXT STEPS:")
if matplotlib_available and seaborn_available:
    print("1. All visualization dependencies are installed")
    print("2. You can now run: python smart_health_ollama.py")
    print("3. The Graphs tab will be fully functional")
else:
    print("1. Install missing dependencies (see above)")
    print("2. Run this test script again to verify")
    print("3. Then run: python smart_health_ollama.py")