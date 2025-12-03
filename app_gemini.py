"""
AI-Powered Receipt OCR System - Streamlit Frontend (Google Gemini)
==================================================================

A web interface for the AI-powered Receipt OCR System using Google Gemini Vision API.

Features:
- True AI-powered receipt analysis (90%+ accuracy)
- Supports any language (Japanese, English, etc.)
- Very cost-effective (~$0.001 per receipt)
- Persistent data storage (images + CSV database)
- Analytics dashboard with visualizations

Author: AI Assistant
Version: 2.1.0 (Gemini AI-Powered)
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# Import the Gemini AI-powered backend
from receipt_ocr_system_gemini import (
    ReceiptOCRSystem,
    ProcessingStatus,
    ExpenseCategory
)

# =============================================================================
# Configuration Constants
# =============================================================================

SAVED_RECEIPTS_DIR = Path("saved_receipts")
RECEIPT_HISTORY_CSV = Path("receipt_history.csv")
CSV_COLUMNS = ["Timestamp", "Date", "Total_Amount", "Currency", "Category", "Merchant", "Image_Path"]
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "gif", "webp"]


# =============================================================================
# Initialization Functions
# =============================================================================

def init_directories():
    """Create necessary directories if they don't exist."""
    SAVED_RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)


def init_csv_database():
    """Initialize the CSV database file with headers if it doesn't exist."""
    if not RECEIPT_HISTORY_CSV.exists():
        with open(RECEIPT_HISTORY_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def get_api_key() -> Optional[str]:
    """Get the API key from session state or environment."""
    if st.session_state.get('api_key'):
        return st.session_state['api_key']
    return os.environ.get('GOOGLE_API_KEY')


@st.cache_resource
def get_ocr_system(_api_key: str) -> ReceiptOCRSystem:
    """Initialize and cache the AI-powered ReceiptOCRSystem."""
    return ReceiptOCRSystem(api_key=_api_key)


# =============================================================================
# Data Persistence Functions
# =============================================================================

def save_uploaded_image(uploaded_file) -> Path:
    """Save an uploaded image to the saved_receipts directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_name = Path(uploaded_file.name).stem
    extension = Path(uploaded_file.name).suffix.lower()
    
    if extension not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        extension = '.jpg'
    
    filename = f"{timestamp}_{original_name}{extension}"
    save_path = SAVED_RECEIPTS_DIR / filename
    
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path


def append_to_csv(
    timestamp: datetime,
    receipt_date: Optional[datetime],
    total_amount: Optional[float],
    currency: str,
    category: str,
    merchant: Optional[str],
    image_path: Path
):
    """Append a new receipt record to the CSV database."""
    with open(RECEIPT_HISTORY_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp.isoformat(),
            receipt_date.strftime("%Y-%m-%d") if receipt_date else "",
            f"{total_amount:.2f}" if total_amount is not None else "",
            currency,
            category,
            merchant or "",
            str(image_path)
        ])


def load_receipt_history() -> pd.DataFrame:
    """Load the receipt history from CSV into a DataFrame."""
    if not RECEIPT_HISTORY_CSV.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    
    try:
        df = pd.read_csv(RECEIPT_HISTORY_CSV, encoding='utf-8')
        
        if 'Timestamp' in df.columns and not df.empty:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        if 'Date' in df.columns and not df.empty:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if 'Total_Amount' in df.columns and not df.empty:
            df['Total_Amount'] = pd.to_numeric(df['Total_Amount'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading receipt history: {e}")
        return pd.DataFrame(columns=CSV_COLUMNS)


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key configuration
        st.subheader("🔑 Google AI API Key")
        
        env_key = os.environ.get('GOOGLE_API_KEY', '')
        
        if env_key:
            st.success("✅ API key found in environment")
            st.session_state['api_key'] = env_key
        else:
            st.caption("Enter your Google AI API key for Gemini Vision.")
            api_key = st.text_input(
                "API Key",
                value=st.session_state.get('api_key', ''),
                type="password",
                placeholder="AIza...",
                help="Get your API key from Google AI Studio"
            )
            st.session_state['api_key'] = api_key
            
            if api_key:
                st.success("✅ API key configured")
            else:
                st.warning("⚠️ API key required")
                st.caption(
                    "Get your free API key from:\n"
                    "[Google AI Studio](https://aistudio.google.com/apikey)"
                )
        
        st.divider()
        
        # AI Model Info
        st.subheader("🤖 AI Model")
        st.info("Using **Gemini 1.5 Flash** for receipt analysis")
        st.caption(
            "✅ 90%+ accuracy\n"
            "✅ Any language support\n"
            "✅ **FREE tier available!**\n"
            "✅ ~$0.001 per receipt (paid)"
        )
        
        st.divider()
        
        # System Info
        st.subheader("📊 System Info")
        
        if SAVED_RECEIPTS_DIR.exists():
            receipt_count = len(list(SAVED_RECEIPTS_DIR.glob("*")))
            st.metric("Saved Images", receipt_count)
        
        if RECEIPT_HISTORY_CSV.exists():
            df = load_receipt_history()
            st.metric("Database Records", len(df))
        else:
            st.metric("Database Records", 0)
        
        st.divider()
        
        # About section
        st.subheader("ℹ️ About")
        st.caption(
            "**AI-Powered Receipt OCR v2.1**\n\n"
            "Powered by Google Gemini Vision:\n"
            "• Accurate text extraction\n"
            "• Auto language detection\n"
            "• Smart categorization\n"
            "• Context understanding"
        )


def render_upload_tab():
    """Render the Upload & Extract tab."""
    st.header("📤 Upload & Extract")
    
    # Check API key
    api_key = get_api_key()
    if not api_key:
        st.error(
            "🔑 **API Key Required**\n\n"
            "Please enter your Google AI API key in the sidebar.\n\n"
            "Get a **FREE** API key from: [Google AI Studio](https://aistudio.google.com/apikey)"
        )
        
        # Show setup instructions
        with st.expander("📖 Setup Instructions", expanded=True):
            st.markdown("""
            ### How to get your FREE Google AI API Key:
            
            1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
            2. Sign in with your Google account
            3. Click "Create API Key"
            4. Copy the key (starts with `AIza...`)
            5. Paste it in the sidebar
            
            ### Free Tier Limits:
            - **15 requests per minute**
            - **1,500 requests per day**
            - Perfect for personal use!
            """)
        return
    
    st.write(
        "Upload a receipt image for AI-powered extraction. "
        "Supports **any language** including Japanese, English, Chinese, etc."
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a receipt image",
        type=SUPPORTED_FORMATS,
        help="Upload a clear image of your receipt"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        
        with col2:
            st.subheader("🤖 AI Extraction Results")
            
            if st.button("🚀 Analyze with Gemini AI", type="primary", use_container_width=True):
                process_receipt_ai(uploaded_file, image, api_key)
            else:
                st.info("Click 'Analyze with Gemini AI' to extract information")


def process_receipt_ai(uploaded_file, image: Image.Image, api_key: str):
    """Process the uploaded receipt using Gemini Vision AI."""
    
    with st.spinner("🤖 Gemini AI is analyzing your receipt..."):
        try:
            ocr_system = get_ocr_system(api_key)
            result = ocr_system.process_image(image)
            
            # Display status
            if result.status == ProcessingStatus.SUCCESS:
                st.success("✅ AI extraction completed successfully!")
            elif result.status == ProcessingStatus.PARTIAL:
                st.warning("⚠️ Partial extraction - some fields could not be identified")
            else:
                st.error("❌ Extraction failed")
            
            # Display extracted data
            if result.data:
                display_extraction_results(result)
                save_receipt_data(uploaded_file, result)
            
            # Display errors
            if result.errors:
                for error in result.errors:
                    st.error(f"Error: {error}")
            
            if result.warnings:
                for warning in result.warnings:
                    st.warning(f"Warning: {warning}")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)


def display_extraction_results(result):
    """Display the extraction results in a formatted manner."""
    data = result.data
    
    # Currency symbol
    currency_symbols = {'JPY': '¥', 'USD': '$', 'EUR': '€', 'GBP': '£', 'CNY': '¥', 'KRW': '₩'}
    currency_symbol = currency_symbols.get(data.currency, data.currency)
    is_jpy = data.currency == 'JPY'
    
    # Language display
    lang_names = {
        'ja': '🇯🇵 Japanese', 'en': '🇺🇸 English', 'zh': '🇨🇳 Chinese', 
        'ko': '🇰🇷 Korean', 'es': '🇪🇸 Spanish', 'fr': '🇫🇷 French'
    }
    lang_display = lang_names.get(data.detected_language, data.detected_language)
    
    # Confidence indicator with color
    conf = data.confidence_score
    if conf >= 0.9:
        conf_emoji = "🎯"
        conf_status = "Excellent"
    elif conf >= 0.7:
        conf_emoji = "✅"
        conf_status = "Good"
    else:
        conf_emoji = "⚠️"
        conf_status = "Low"
    
    # Key metrics
    metric_cols = st.columns(3)
    
    with metric_cols[0]:
        if data.date:
            if data.detected_language == 'ja':
                date_str = data.date.strftime("%Y年%m月%d日")
            else:
                date_str = data.date.strftime("%B %d, %Y")
        else:
            date_str = "Not found"
        st.metric("📅 Date", date_str)
    
    with metric_cols[1]:
        if data.total_amount:
            if is_jpy:
                total_str = f"{currency_symbol}{data.total_amount:,.0f}"
            else:
                total_str = f"{currency_symbol}{data.total_amount:.2f}"
        else:
            total_str = "Not found"
        st.metric("💰 Total", total_str)
    
    with metric_cols[2]:
        st.metric("🏷️ Category", data.category.value)
    
    # Confidence score display
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #4CAF50{int(conf*100):02x} {conf*100}%, #f0f0f0 {conf*100}%); 
                padding: 12px 16px; border-radius: 8px; margin: 10px 0; border: 1px solid #ddd;">
        <span style="font-size: 16px;">{conf_emoji} <b>AI Confidence: {conf:.1%}</b> ({conf_status})</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional details
    with st.expander("📋 Full Details", expanded=True):
        detail_cols = st.columns(2)
        
        with detail_cols[0]:
            st.write("**🏪 Merchant:**", data.merchant_name or "Not found")
            st.write("**📍 Address:**", data.merchant_address or "Not found")
            st.write("**📞 Phone:**", data.merchant_phone or "Not found")
            st.write("**💳 Payment:**", data.payment_method or "Not found")
        
        with detail_cols[1]:
            if data.subtotal:
                subtotal_str = f"{currency_symbol}{data.subtotal:,.0f}" if is_jpy else f"{currency_symbol}{data.subtotal:.2f}"
            else:
                subtotal_str = "Not found"
            st.write("**📝 Subtotal:**", subtotal_str)
            
            if data.tax_amount:
                tax_str = f"{currency_symbol}{data.tax_amount:,.0f}" if is_jpy else f"{currency_symbol}{data.tax_amount:.2f}"
                if data.tax_rate:
                    tax_str += f" ({data.tax_rate})"
            else:
                tax_str = "Not found"
            st.write("**📊 Tax:**", tax_str)
            
            if data.discount_amount:
                disc_str = f"-{currency_symbol}{data.discount_amount:,.0f}" if is_jpy else f"-{currency_symbol}{data.discount_amount:.2f}"
                st.write("**🏷️ Discount:**", disc_str)
            
            st.write("**🕐 Time:**", data.time or "Not found")
            st.write("**🌐 Language:**", lang_display)
            st.write("**💱 Currency:**", data.currency)
        
        st.write("**⏱️ Processing Time:**", f"{result.processing_time_ms:.0f}ms")
    
    # Line items
    if data.line_items:
        with st.expander(f"🛒 Line Items ({len(data.line_items)})", expanded=True):
            items_data = []
            for item in data.line_items:
                price_str = f"{currency_symbol}{item.total_price:,.0f}" if is_jpy else f"{currency_symbol}{item.total_price:.2f}"
                discount_str = ""
                if item.discount and item.discount > 0:
                    discount_str = f" (-{currency_symbol}{item.discount:,.0f})" if is_jpy else f" (-{currency_symbol}{item.discount:.2f})"
                items_data.append({
                    "Item": item.description,
                    "Qty": int(item.quantity) if item.quantity == int(item.quantity) else item.quantity,
                    "Price": price_str + discount_str
                })
            
            st.dataframe(
                pd.DataFrame(items_data),
                use_container_width=True,
                hide_index=True
            )
    
    # AI Notes
    if data.ai_notes:
        with st.expander("🤖 AI Notes", expanded=False):
            st.info(data.ai_notes)
    
    # Raw AI response
    with st.expander("📝 Raw AI Response (JSON)", expanded=False):
        st.code(data.raw_text, language='json')


def save_receipt_data(uploaded_file, result):
    """Save the processed receipt data to persistent storage."""
    try:
        uploaded_file.seek(0)
        image_path = save_uploaded_image(uploaded_file)
        
        timestamp = datetime.now()
        append_to_csv(
            timestamp=timestamp,
            receipt_date=result.data.date,
            total_amount=result.data.total_amount,
            currency=result.data.currency,
            category=result.data.category.value,
            merchant=result.data.merchant_name,
            image_path=image_path
        )
        
        st.success(f"💾 Receipt saved to database!")
        
    except Exception as e:
        st.error(f"Failed to save receipt: {e}")


def render_analytics_tab():
    """Render the Analytics Dashboard tab."""
    st.header("📊 Analytics Dashboard")
    
    df = load_receipt_history()
    
    if df.empty:
        st.info(
            "📭 No receipt data yet!\n\n"
            "Upload and process some receipts in the 'Upload & Extract' tab "
            "to see analytics here."
        )
        return
    
    df_with_totals = df[df['Total_Amount'].notna()].copy()
    
    # Summary Metrics
    st.subheader("📈 Summary Metrics")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        total_spent = df_with_totals['Total_Amount'].sum()
        st.metric("💰 Total Spent", f"¥{total_spent:,.0f}")
    
    with metric_cols[1]:
        avg_transaction = df_with_totals['Total_Amount'].mean()
        st.metric("📊 Average", f"¥{avg_transaction:,.0f}" if not pd.isna(avg_transaction) else "¥0")
    
    with metric_cols[2]:
        st.metric("🧾 Total Receipts", len(df))
    
    with metric_cols[3]:
        unique_merchants = df['Merchant'].nunique()
        st.metric("🏪 Unique Stores", unique_merchants)
    
    st.divider()
    
    # Visualizations
    st.subheader("📉 Visualizations")
    
    viz_cols = st.columns(2)
    
    with viz_cols[0]:
        st.write("**Spending by Category**")
        if not df_with_totals.empty:
            category_spending = df_with_totals.groupby('Category')['Total_Amount'].sum().reset_index()
            fig_pie = px.pie(
                category_spending,
                values='Total_Amount',
                names='Category',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with viz_cols[1]:
        st.write("**Spending Over Time**")
        if not df_with_totals.empty and 'Timestamp' in df_with_totals.columns:
            df_time = df_with_totals.copy()
            df_time['Scan_Date'] = df_time['Timestamp'].dt.date
            daily_spending = df_time.groupby('Scan_Date')['Total_Amount'].sum().reset_index()
            daily_spending['Scan_Date'] = pd.to_datetime(daily_spending['Scan_Date'])
            
            fig_bar = px.bar(
                daily_spending,
                x='Scan_Date',
                y='Total_Amount',
                labels={'Total_Amount': 'Amount (¥)', 'Scan_Date': 'Date'},
                color_discrete_sequence=['#2196F3']
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Top Merchants
    if 'Merchant' in df_with_totals.columns:
        st.subheader("🏪 Top Merchants")
        merchant_spending = df_with_totals.groupby('Merchant')['Total_Amount'].agg(['sum', 'count']).reset_index()
        merchant_spending.columns = ['Merchant', 'Total Spent', 'Visits']
        merchant_spending = merchant_spending.sort_values('Total Spent', ascending=False).head(10)
        merchant_spending['Total Spent'] = merchant_spending['Total Spent'].apply(lambda x: f"¥{x:,.0f}")
        st.dataframe(merchant_spending, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # History Table
    st.subheader("📜 Receipt History")
    
    display_df = df.copy()
    if 'Timestamp' in display_df.columns:
        display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    if 'Date' in display_df.columns:
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Date'] = display_df['Date'].fillna('N/A')
    if 'Total_Amount' in display_df.columns:
        display_df['Total_Amount'] = display_df['Total_Amount'].apply(
            lambda x: f"¥{x:,.0f}" if pd.notna(x) else "N/A"
        )
    
    # Remove Image_Path from display
    if 'Image_Path' in display_df.columns:
        display_df = display_df.drop(columns=['Image_Path'])
    
    display_df = display_df.iloc[::-1].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.download_button(
        label="📥 Download History (CSV)",
        data=df.to_csv(index=False),
        file_name=f"receipt_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Gemini Receipt OCR",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_directories()
    init_csv_database()
    
    # Custom CSS
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 50px; padding: 0 20px; }
        div[data-testid="stMetricValue"] { font-size: 24px; }
        </style>
    """, unsafe_allow_html=True)
    
    # Title with Gemini branding
    st.title("🤖 AI Receipt OCR (Gemini)")
    st.caption("Powered by Google Gemini Vision • 90%+ accuracy • Any language • **FREE tier available!**")
    
    render_sidebar()
    
    # Tabs
    tab1, tab2 = st.tabs(["📤 Upload & Extract", "📊 Analytics Dashboard"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_analytics_tab()
    
    # Footer
    st.divider()
    st.caption(
        "Built with ❤️ using Google Gemini Vision API, Streamlit, and Plotly | "
        "AI Receipt OCR System v2.1"
    )


if __name__ == "__main__":
    main()
