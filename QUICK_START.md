# Quick Start Guide - Gemini Receipt OCR

## ✅ Environment Status

Your environment is **fully set up** and ready to use!

## 🚀 Running the App

### Step 1: Get Your API Key (if not already done)

1. Visit: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (starts with `AIza...`)

### Step 2: Set API Key (Choose one method)

**Option A: Environment Variable (Recommended)**
```powershell
# PowerShell
$env:GOOGLE_API_KEY="AIza...your-key-here"

# Command Prompt
set GOOGLE_API_KEY=AIza...your-key-here
```

**Option B: Enter in App**
- Just run the app and enter the API key in the sidebar

### Step 3: Run the App

```bash
python -m streamlit run app_gemini.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📋 What's Installed

✅ **All Required Packages:**
- streamlit (web framework)
- pandas (data analysis)
- plotly (visualizations)
- Pillow (image processing)
- google-generativeai (Gemini AI)
- numpy (numerical computing)

✅ **Project Files:**
- `app_gemini.py` - Main Streamlit application
- `receipt_ocr_system_gemini.py` - AI backend
- `saved_receipts/` - Folder for receipt images
- `receipt_history.csv` - Database of processed receipts

## 🔍 Verify Setup

Run this anytime to check your setup:
```bash
python verify_setup.py
```

## 💡 Usage Tips

1. **Upload Receipts**: Go to "Upload & Extract" tab
2. **View Analytics**: Go to "Analytics Dashboard" tab
3. **Download Data**: Click "Download History (CSV)" button

## 🆘 Troubleshooting

**Port already in use?**
- Streamlit will automatically use the next available port
- Or specify: `streamlit run app_gemini.py --server.port 8502`

**Import errors?**
- Run: `pip install streamlit pandas plotly Pillow google-generativeai`

**API key issues?**
- Make sure it starts with `AIza...`
- Check for extra spaces when copying
- Try entering it in the app sidebar instead

## 📚 Free Tier Limits

- 15 requests per minute
- 1,500 requests per day
- Perfect for personal use!

Enjoy your receipt OCR system! 🎉

