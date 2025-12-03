#!/usr/bin/env python3
"""
Setup Verification Script for Gemini Receipt OCR
Checks if all dependencies and environment are properly configured.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported."""
    print("🔍 Checking Python package imports...")
    
    packages = {
        'streamlit': 'Streamlit web framework',
        'pandas': 'Data analysis library',
        'plotly': 'Visualization library',
        'PIL': 'Image processing (Pillow)',
        'google.generativeai': 'Google Gemini AI',
        'numpy': 'Numerical computing'
    }
    
    failed = []
    for package, description in packages.items():
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"  ✅ {package:20} - {description}")
        except ImportError as e:
            print(f"  ❌ {package:20} - MISSING! ({e})")
            failed.append(package)
    
    return len(failed) == 0

def check_local_modules():
    """Check if local modules can be imported."""
    print("\n🔍 Checking local modules...")
    
    try:
        from receipt_ocr_system_gemini import ReceiptOCRSystem, ProcessingStatus, ExpenseCategory
        print("  ✅ receipt_ocr_system_gemini.py - OK")
        return True
    except ImportError as e:
        print(f"  ❌ receipt_ocr_system_gemini.py - ERROR: {e}")
        return False

def check_directories():
    """Check if required directories exist."""
    print("\n🔍 Checking directories...")
    
    required_dirs = ['saved_receipts']
    all_ok = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✅ {dir_name}/ - exists")
        else:
            print(f"  ⚠️  {dir_name}/ - will be created automatically")
            all_ok = False
    
    return True  # Directories are created automatically, so this is OK

def check_api_key():
    """Check if API key is configured."""
    print("\n🔍 Checking API key configuration...")
    
    import os
    api_key = os.environ.get('GOOGLE_API_KEY')
    
    if api_key:
        masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
        print(f"  ✅ GOOGLE_API_KEY environment variable is set ({masked_key})")
        return True
    else:
        print("  ⚠️  GOOGLE_API_KEY not set in environment")
        print("     You can set it with: set GOOGLE_API_KEY=your-key-here")
        print("     Or enter it in the app sidebar when running")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Gemini Receipt OCR - Environment Verification")
    print("=" * 60)
    
    results = {
        'imports': check_imports(),
        'modules': check_local_modules(),
        'directories': check_directories(),
        'api_key': check_api_key()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all([results['imports'], results['modules'], results['directories']]):
        print("✅ Environment setup is COMPLETE!")
        print("\n📝 Next steps:")
        print("   1. Get your Google AI API key from: https://aistudio.google.com/apikey")
        if not results['api_key']:
            print("   2. Set API key: set GOOGLE_API_KEY=your-key-here")
        print("   3. Run the app: python -m streamlit run app_gemini.py")
        return 0
    else:
        print("❌ Some issues found. Please fix them before running the app.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

