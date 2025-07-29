#!/usr/bin/env python3
"""
Setup script to install all required dependencies for the Smart Health Agent
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_installation():
    """Check if all required packages are installed"""
    print("\n🔍 Checking installed packages...")
    
    required_packages = [
        "gradio", "langchain", "langchain-core", "langchain-community", 
        "langchain-ollama", "pandas", "numpy", "requests", "pydantic",
        "matplotlib", "seaborn", "ollama", "sentence-transformers",
        "transformers", "torch", "pymilvus", "langgraph", "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    return missing_packages

def main():
    print("🏥 Smart Health Agent - Dependency Setup")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        print("   Please run this script from the smart_health_agent directory")
        return 1
    
    # Install from requirements.txt
    print("📋 Installing dependencies from requirements.txt...")
    success = run_command("pip install -r requirements.txt", "Installing core dependencies")
    
    if not success:
        print("\n⚠️ Installation may have failed. Trying alternative methods...")
        
        # Try with python -m pip
        success = run_command("python -m pip install -r requirements.txt", "Installing with python -m pip")
        
        if not success:
            # Try with pip3
            success = run_command("pip3 install -r requirements.txt", "Installing with pip3")
    
    # Check what's installed
    missing = check_installation()
    
    if missing:
        print(f"\n⚠️ Some packages are still missing: {', '.join(missing)}")
        print("   You may need to install them manually:")
        print(f"   pip install {' '.join(missing)}")
    else:
        print("\n✅ All required packages are installed!")
    
    # Test key functionality
    print("\n🧪 Testing key functionality...")
    
    try:
        import database
        database.create_tables()
        print("✅ Database functionality working")
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    
    try:
        import gradio as gr
        print("✅ Gradio UI framework working")
    except Exception as e:
        print(f"❌ Gradio test failed: {e}")
    
    try:
        import health_visualizations
        print("✅ Visualization features working")
    except Exception as e:
        print(f"ℹ️ Visualization features not available: {e}")
        print("   This is normal if matplotlib/seaborn installation failed")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run: python run_health_agent.py")
    print("2. Or: python smart_health_ollama.py")
    print("3. If visualizations aren't working, manually install:")
    print("   pip install matplotlib seaborn")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())