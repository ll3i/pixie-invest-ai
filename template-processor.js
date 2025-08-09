// Flask 템플릿 문법을 간단히 처리하는 JavaScript
function processFlaskTemplate(html) {
    // layout.html의 기본 구조
    const layoutHTML = `
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixie - AI 투자 어드바이저</title>
    <link href="https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="static/css/style.css">
    <link rel="stylesheet" href="static/css/chatbot-widget.css">
    <style>
        :root {
            --primary-color: #1454FE;
            --primary-dark: #0B3DCF;
            --primary-light: #E6EFFF;
            --secondary-color: #6C7B95;
            --success-color: #22C55E;
            --danger-color: #EF4444;
            --warning-color: #F59E0B;
            --dark-color: #1F2937;
            --light-color: #F8FAFC;
            --border-color: #E2E8F0;
            --text-primary: #0F172A;
            --text-secondary: #64748B;
        }
        body {
            font-family: 'Pretendard', sans-serif;
        }
    </style>
    EXTRA_CSS_PLACEHOLDER
</head>
<body>
    CONTENT_PLACEHOLDER
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    EXTRA_JS_PLACEHOLDER
</body>
</html>`;

    // extends 'layout.html' 처리
    if (html.includes("{% extends 'layout.html' %}")) {
        // 블록 내용 추출
        let extraCSS = '';
        let content = '';
        let extraJS = '';
        
        // extra_css 블록 추출
        const cssMatch = html.match(/{% block extra_css %}([\s\S]*?){% endblock %}/);
        if (cssMatch) {
            extraCSS = cssMatch[1];
        }
        
        // content 블록 추출
        const contentMatch = html.match(/{% block content %}([\s\S]*?){% endblock %}/);
        if (contentMatch) {
            content = contentMatch[1];
        }
        
        // extra_js 블록 추출
        const jsMatch = html.match(/{% block extra_js %}([\s\S]*?){% endblock %}/);
        if (jsMatch) {
            extraJS = jsMatch[1];
        }
        
        // layout에 내용 삽입
        html = layoutHTML
            .replace('EXTRA_CSS_PLACEHOLDER', extraCSS)
            .replace('CONTENT_PLACEHOLDER', content)
            .replace('EXTRA_JS_PLACEHOLDER', extraJS);
    }
    
    // url_for 처리
    html = html.replace(/{{ url_for\('static', filename='([^']+)'\)[^}]*}}/g, 'static/$1');
    
    // 간단한 변수 처리 (기본값으로 대체)
    html = html.replace(/{{ [^}]+ }}/g, function(match) {
        if (match.includes('current_user')) return '사용자';
        if (match.includes('user_name')) return '투자자';
        if (match.includes('date')) return new Date().toLocaleDateString('ko-KR');
        if (match.includes('time')) return new Date().toLocaleTimeString('ko-KR');
        return '';
    });
    
    // if 문 처리 (간단히 true로 가정)
    html = html.replace(/{% if [^%]+ %}/g, '');
    html = html.replace(/{% else %}/g, '<!--');
    html = html.replace(/{% endif %}/g, '-->');
    
    // for 문 제거 (샘플 데이터로 대체 가능)
    html = html.replace(/{% for [^%]+ %}[\s\S]*?{% endfor %}/g, '<!-- 반복 컨텐츠 -->');
    
    // 나머지 템플릿 태그 제거
    html = html.replace(/{%[^%]*%}/g, '');
    
    return html;
}

// 페이지 로드 시 템플릿 처리
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        const bodyHTML = document.body.innerHTML;
        if (bodyHTML.includes('{%') || bodyHTML.includes('{{')) {
            document.body.innerHTML = processFlaskTemplate(bodyHTML);
        }
    });
} else {
    const bodyHTML = document.body.innerHTML;
    if (bodyHTML.includes('{%') || bodyHTML.includes('{{')) {
        document.body.innerHTML = processFlaskTemplate(bodyHTML);
    }
}