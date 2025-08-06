# Pixie 프로젝트 필수 파일 복사 스크립트

$sourceBase = "C:\Users\work4\OneDrive\바탕 화면\투자챗봇"
$destBase = "C:\Users\work4\OneDrive\바탕 화면\kb ai challenge\code"

Write-Host "Pixie 프로젝트 필수 파일 복사 시작..." -ForegroundColor Green

# src 디렉토리 파일들
Write-Host "`n[1/5] src 디렉토리 파일 복사 중..." -ForegroundColor Yellow
$srcFiles = @(
    "main.py", "investment_advisor.py", "config.py", "llm_service.py",
    "db_client.py", "financial_data_processor.py", "data_collector.py",
    "user_profile_analyzer.py", "memory_manager.py", "simplified_portfolio_prediction.py",
    "advanced_stock_predictor.py", "stock_search_engine.py", "financial_report_analyzer.py",
    "news_collector_service.py", "user_survey.py", "survey_to_profile.py",
    "recommend_portfolio.py", "prompt_AI-A.txt", "prompt_AI-A2.txt", "prompt_AI-B.txt",
    "prompt_manager.py", "data_processor.py", "stock_evaluator.py",
    "korean_stock_data_processor.py", "api_service.py", "data_update_scheduler.py",
    "response_filter.py", "prompt_survey-analysis.txt", "prompt_survey-score.txt"
)

foreach ($file in $srcFiles) {
    $source = Join-Path $sourceBase "src\$file"
    $dest = Join-Path $destBase "src\$file"
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $dest -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $file (not found)" -ForegroundColor Red
    }
}

# web 디렉토리 메인 파일들
Write-Host "`n[2/5] web 디렉토리 파일 복사 중..." -ForegroundColor Yellow
$webFiles = @("app.py", "news_api_helper.py", "config.py", "requirements.txt")
foreach ($file in $webFiles) {
    $source = Join-Path $sourceBase "web\$file"
    $dest = Join-Path $destBase "web\$file"
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $dest -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    }
}

# templates 디렉토리
Write-Host "`n[3/5] templates 디렉토리 복사 중..." -ForegroundColor Yellow
$templateFiles = @(
    "layout.html", "index.html", "dashboard.html", "survey.html",
    "survey_result.html", "result.html", "learning.html", "learning_terms.html",
    "learning_term.html", "learning_quiz.html", "learning_cardnews.html",
    "chatbot.html", "stock.html", "time_series_prediction.html", "news.html",
    "news_enhanced.html", "my-investment.html", "my_invest.html", "alerts.html",
    "alert_history.html", "market_sentiment.html", "minerva.html"
)

foreach ($file in $templateFiles) {
    $source = Join-Path $sourceBase "web\templates\$file"
    $dest = Join-Path $destBase "web\templates\$file"
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $dest -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    }
}

# static 디렉토리 전체 복사
Write-Host "`n[4/5] static 디렉토리 복사 중..." -ForegroundColor Yellow
$staticSource = Join-Path $sourceBase "web\static"
$staticDest = Join-Path $destBase "web\static"
if (Test-Path $staticSource) {
    Copy-Item -Path $staticSource -Destination $staticDest -Recurse -Force
    Write-Host "  ✓ static 디렉토리 전체 복사 완료" -ForegroundColor Green
}

# services 디렉토리 복사
$servicesSource = Join-Path $sourceBase "web\services"
$servicesDest = Join-Path $destBase "web\services"
if (Test-Path $servicesSource) {
    Copy-Item -Path $servicesSource -Destination $servicesDest -Recurse -Force
    Write-Host "  ✓ services 디렉토리 전체 복사 완료" -ForegroundColor Green
}

# 루트 파일들
Write-Host "`n[5/5] 루트 파일 복사 중..." -ForegroundColor Yellow
$rootFiles = @("start_web_server.bat", "requirements.txt", "CLAUDE.md")
foreach ($file in $rootFiles) {
    $source = Join-Path $sourceBase $file
    $dest = Join-Path $destBase $file
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $dest -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    }
}

# tests 디렉토리
$testFile = "tests\test_investment_advisor.py"
$source = Join-Path $sourceBase $testFile
$dest = Join-Path $destBase $testFile
if (Test-Path $source) {
    Copy-Item -Path $source -Destination $dest -Force
    Write-Host "  ✓ $testFile" -ForegroundColor Green
}

# data 디렉토리 구조 생성
New-Item -ItemType Directory -Path "$destBase\data\raw" -Force | Out-Null
New-Item -ItemType Directory -Path "$destBase\data\processed" -Force | Out-Null
New-Item -ItemType Directory -Path "$destBase\docs" -Force | Out-Null

Write-Host "`n✅ 복사 완료!" -ForegroundColor Green
Write-Host "대상 디렉토리: $destBase" -ForegroundColor Cyan

# 복사된 파일 수 확인
$copiedFiles = Get-ChildItem -Path $destBase -Recurse -File
Write-Host "`n총 복사된 파일 수: $($copiedFiles.Count)개" -ForegroundColor Yellow