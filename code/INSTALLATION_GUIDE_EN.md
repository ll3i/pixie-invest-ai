# Pixie - AI-Powered Personalized Investment Advisory System Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start Guide](#quick-start-guide)
3. [Detailed Installation](#detailed-installation)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Features Overview](#features-overview)

## System Requirements

### Required
- **Python**: 3.8 or higher (3.9 - 3.11 recommended)
- **Operating System**: Windows 10/11, macOS, Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: At least 1GB free space
- **Internet**: Stable connection for API communication

### Required API Keys
- **OpenAI API Key**: Required for ChatGPT-based AI investment advice
  - Get it from: https://platform.openai.com/api-keys
- **Flask Secret Key**: Required for web session security (32+ characters)

### Optional API Keys
- **Supabase**: For cloud database usage (optional)
- **Clova Studio**: Alternative LLM to OpenAI (optional)

## Quick Start Guide

### For Windows Users (5-minute setup)

1. **Download the project**
   ```bash
   git clone [GitHub URL]
   cd pixie
   ```

2. **Create environment configuration**
   ```bash
   copy .env.example .env
   notepad .env
   ```
   Enter required values:
   ```
   OPENAI_API_KEY=sk-your-openai-api-key
   FLASK_SECRET_KEY=your-32-character-secret-key-here!!!
   ```

3. **Automatic setup and run**
   ```bash
   setup_and_run.bat
   ```

4. **Open in web browser**
   ```
   http://localhost:5000
   ```

### For macOS/Linux Users

1. **Download the project**
   ```bash
   git clone [GitHub URL]
   cd pixie
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   nano .env  # or vim .env
   ```

5. **Run the application**
   ```bash
   cd web
   python app.py
   ```

## Detailed Installation

### Step 1: Verify Python Installation
```bash
python --version
# Should be Python 3.8.0 or higher
```

If Python is not installed:
- Windows: https://www.python.org/downloads/
- macOS: `brew install python3`
- Linux: `sudo apt-get install python3 python3-pip`

### Step 2: Download the Project
```bash
# Using Git
git clone [GitHub URL]
cd pixie

# Or download and extract ZIP file
```

### Step 3: Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- Flask (web framework)
- pandas (data processing)
- scikit-learn (machine learning)
- openai (AI API)
- yfinance (stock data)
- statsmodels (time series analysis)

### Step 5: Configure Environment Variables
1. Copy `.env.example` to `.env`
2. Open `.env` file in a text editor
3. Enter required API keys:

```env
# Required settings
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
FLASK_SECRET_KEY=enter-32-or-more-random-characters-here!!!

# Optional settings (comment out if not using)
# SUPABASE_URL=https://xxxxx.supabase.co
# SUPABASE_KEY=eyJxxxxxxxxxx
# CLOVA_API_KEY=xxxxxxxxxxxxx
```

### Step 6: Initialize Database
The database is created automatically on first run.
To manually initialize:
```bash
python web/app.py --init-db
```

## Configuration

### API Key Setup Guide

#### OpenAI API Key
1. Visit https://platform.openai.com
2. Sign in or create an account
3. Click on API Keys menu
4. Click "Create new secret key"
5. Copy the generated key to your `.env` file

#### Generate Flask Secret Key
Generate a secure key using Python:
```python
import secrets
print(secrets.token_hex(32))
```

### Port Configuration (Optional)
To change the default port (5000), edit `web/app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Change to desired port
```

## Running the Application

### Method 1: Using Batch Files (Windows)
```bash
# Quick start
run_pixie_web.bat

# Full setup and run
setup_and_run.bat
```

### Method 2: Command Line
```bash
# From project root
python web/app.py

# Or from web directory
cd web
python app.py
```

### Method 3: Development Mode
```bash
# Enable debug mode
export FLASK_ENV=development  # Linux/macOS
set FLASK_ENV=development     # Windows
python web/app.py
```

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError
```
Error: ModuleNotFoundError: No module named 'openai'
Solution: pip install -r requirements.txt
```

#### 2. ImportError: Cannot find src module
```
Error: ImportError: cannot import name 'InvestmentAdvisor'
Solution: Run from project root directory or set PYTHONPATH
```

#### 3. OpenAI API Error
```
Error: openai.error.AuthenticationError
Solution: Check OPENAI_API_KEY in .env file
```

#### 4. Port Already in Use
```
Error: Address already in use
Solution: Use different port or kill existing process
```

#### 5. Database Error
```
Error: sqlite3.OperationalError
Solution: Check permissions for web/investment_data.db
```

### Checking Logs
Check log files when issues occur:
- `minerva_YYYYMMDD.log`: System-wide logs
- `web/api_service.log`: API call logs
- `data_update.log`: Data update logs

## Features Overview

### 1. Investment Survey (10 questions)
- Investment preference analysis
- Personalized portfolio recommendations
- URL: `/survey`

### 2. AI Chatbot
- Real-time investment consultation
- Multi-agent AI system
- URL: `/chatbot`

### 3. Stock Prediction
- ARIMA-X model based
- Korean/US stock support
- URL: `/predictions`

### 4. News/Issues
- Real-time financial news
- Personalized keyword filtering
- URL: `/news`

### 5. Investment Learning
- Investment basics education
- Terminology explanations
- URL: `/learning`

### 6. MY Investment
- Portfolio management
- Investment tracking
- URL: `/my-investment`

## Additional Information

### Data Updates
```bash
# Update all data
python src/main.py --update-data all

# Update news only
python src/main.py --update-data news
```

### Performance Optimization
- Set `FLASK_ENV=production` for production environment
- Use WSGI server like Gunicorn recommended
- Regular database optimization

### Security Notes
- Never commit `.env` file to Git
- Regularly rotate Flask Secret Key
- Keep API keys secure

## Support

If you encounter issues during installation:
1. Check [GitHub Issues] page
2. Refer to technical documentation in `CLAUDE.md`
3. Check log files and search for error messages

---

**Version**: 1.0.0  
**Last Updated**: January 2024  
**License**: MIT License