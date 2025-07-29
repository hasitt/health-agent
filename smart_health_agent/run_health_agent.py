#!/usr/bin/env python3
"""
Wrapper script to run the Smart Health Agent with better error handling
"""

import sys
import os

def main():
    print("🏥 Smart Health Agent - Enhanced Version")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return 1
    
    # Check if we're in the right directory
    if not os.path.exists("smart_health_ollama.py"):
        print("❌ Please run this script from the smart_health_agent directory")
        return 1
    
    # Test core imports
    print("🔍 Checking dependencies...")
    
    try:
        import gradio
        print("✅ Gradio available")
    except ImportError:
        print("❌ Gradio not installed: pip install gradio")
        return 1
    
    try:
        import database
        print("✅ Database module available")
    except ImportError:
        print("❌ Database module not found")
        return 1
    
    # Check visualization dependencies
    try:
        import matplotlib
        import seaborn
        print("✅ Visualization libraries available")
        viz_available = True
    except ImportError:
        print("ℹ️ Visualization libraries not installed")
        print("   To enable graphs: pip install matplotlib seaborn")
        viz_available = False
    
    # Initialize database
    print("\n🗄️ Initializing database...")
    try:
        import database
        database.create_tables()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return 1
    
    # Start the application
    print("\n🚀 Starting Smart Health Agent...")
    print("   Features available:")
    print("   ✅ Garmin data sync")
    print("   ✅ Cronometer CSV import")
    print("   ✅ Daily mood tracking")
    print("   ✅ Trend analysis")
    print("   ✅ AI health insights")
    
    if viz_available:
        print("   ✅ Data visualizations")
    else:
        print("   ⚠️ Data visualizations (install matplotlib seaborn)")
    
    print("\n🌐 Opening web interface...")
    print("   The application will open in your browser automatically")
    print("   If it doesn't, navigate to: http://localhost:7860")
    print("\n   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the main application
        import smart_health_ollama
        smart_health_ollama.create_ui()
        
    except KeyboardInterrupt:
        print("\n\n👋 Smart Health Agent stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("\nFor detailed error information, run:")
        print("   python smart_health_ollama.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())