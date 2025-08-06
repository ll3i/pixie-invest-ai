@echo off
echo Copying essential files for Pixie project...

set SOURCE_DIR=C:\Users\work4\OneDrive\바탕 화면\투자챗봇
set DEST_DIR=C:\Users\work4\OneDrive\바탕 화면\kb ai challenge\code

echo.
echo [1/4] Copying src files...
copy "%SOURCE_DIR%\src\main.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\investment_advisor.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\config.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\llm_service.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\db_client.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\financial_data_processor.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\data_collector.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\user_profile_analyzer.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\memory_manager.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\simplified_portfolio_prediction.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\advanced_stock_predictor.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\stock_search_engine.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\financial_report_analyzer.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\news_collector_service.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\user_survey.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\survey_to_profile.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\recommend_portfolio.py" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\prompt_AI-A.txt" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\prompt_AI-A2.txt" "%DEST_DIR%\src\" >nul 2>&1
copy "%SOURCE_DIR%\src\prompt_AI-B.txt" "%DEST_DIR%\src\" >nul 2>&1

echo [2/4] Copying web app files...
copy "%SOURCE_DIR%\web\app.py" "%DEST_DIR%\web\" >nul 2>&1
copy "%SOURCE_DIR%\web\news_api_helper.py" "%DEST_DIR%\web\" >nul 2>&1
copy "%SOURCE_DIR%\web\config.py" "%DEST_DIR%\web\" >nul 2>&1
copy "%SOURCE_DIR%\web\requirements.txt" "%DEST_DIR%\web\" >nul 2>&1

echo [3/4] Copying templates...
copy "%SOURCE_DIR%\web\templates\layout.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\index.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\dashboard.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\survey.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\survey_result.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\result.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\learning.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\learning_terms.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\learning_term.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\learning_quiz.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\learning_cardnews.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\chatbot.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\stock.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\time_series_prediction.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\news.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\news_enhanced.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\my-investment.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\my_invest.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\alerts.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\alert_history.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\market_sentiment.html" "%DEST_DIR%\web\templates\" >nul 2>&1
copy "%SOURCE_DIR%\web\templates\minerva.html" "%DEST_DIR%\web\templates\" >nul 2>&1

echo [4/4] Copying static files and services...
xcopy "%SOURCE_DIR%\web\static\*" "%DEST_DIR%\web\static\" /E /Y /Q >nul 2>&1
xcopy "%SOURCE_DIR%\web\services\*" "%DEST_DIR%\web\services\" /E /Y /Q >nul 2>&1

echo.
echo Copying root files...
copy "%SOURCE_DIR%\start_web_server.bat" "%DEST_DIR%\" >nul 2>&1
copy "%SOURCE_DIR%\requirements.txt" "%DEST_DIR%\" >nul 2>&1
copy "%SOURCE_DIR%\CLAUDE.md" "%DEST_DIR%\" >nul 2>&1
copy "%SOURCE_DIR%\README.md" "%DEST_DIR%\" >nul 2>&1

echo.
echo Copying test files...
copy "%SOURCE_DIR%\tests\test_investment_advisor.py" "%DEST_DIR%\tests\" >nul 2>&1

echo.
echo Done! All essential files have been copied to: %DEST_DIR%
echo.
pause