#!/usr/bin/env python3
"""
Setup script for Garmin Connect integration.
This script helps users securely configure their Garmin credentials.
"""

import os
import getpass
from pathlib import Path

def setup_garmin_credentials():
    """
    Interactive setup for Garmin Connect credentials.
    """
    print("🔐 Garmin Connect Credentials Setup")
    print("=" * 40)
    print()
    
    # Check if .env file already exists
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    print("Please enter your Garmin Connect credentials:")
    print("(Your password will be hidden when typing)")
    print()
    
    # Get credentials
    username = input("Garmin Connect Email: ").strip()
    password = getpass.getpass("Garmin Connect Password: ")
    
    if not username or not password:
        print("❌ Error: Both username and password are required.")
        return
    
    # Create .env file
    env_content = f"""# Garmin Connect Credentials
GARMIN_USERNAME={username}
GARMIN_PASSWORD={password}

# Optional: Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Optional: Google Fit (if using)
# GOOGLE_FIT_CLIENT_SECRETS=/path/to/your/client_secrets.json
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print()
        print("✅ Successfully created .env file!")
        print(f"📁 File location: {env_file.absolute()}")
        print()
        print("🔒 Security notes:")
        print("   - The .env file is automatically ignored by git")
        print("   - Never commit this file to version control")
        print("   - Keep your credentials secure")
        print()
        print("🚀 You can now use the Garmin integration!")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def test_garmin_connection():
    """
    Test the Garmin connection with the configured credentials.
    """
    print("🧪 Testing Garmin Connect connection...")
    print()
    
    try:
        from garmin_utils import get_garmin_health_data
        
        # Test the connection
        health_data = get_garmin_health_data()
        
        print("✅ Connection successful!")
        print()
        print("📊 Sample data retrieved:")
        print(f"   Steps: {health_data['steps']['steps']}")
        print(f"   Sleep: {health_data['sleep']['sleep_hours']} hours")
        print(f"   Resting HR: {health_data['heart_rate']['resting_hr']} bpm")
        print()
        print("🎉 Garmin integration is working correctly!")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Please check your credentials and try again.")

def main():
    """
    Main setup function.
    """
    print("🏃‍♂️ Smart Health Agent - Garmin Setup")
    print("=" * 50)
    print()
    
    while True:
        print("Choose an option:")
        print("1. Setup Garmin credentials")
        print("2. Test Garmin connection")
        print("3. Exit")
        print()
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            setup_garmin_credentials()
        elif choice == '2':
            test_garmin_connection()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        print()
        input("Press Enter to continue...")
        print()

if __name__ == "__main__":
    main() 