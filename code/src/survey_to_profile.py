import json
from datetime import datetime
from db_client import get_supabase_client

# (1) LLM 분석 함수 예시 (OpenAI API 사용)
def analyze_survey_with_llm(survey_json):
    import openai
    prompt = f"""
    다음은 투자 성향 분석 설문 결과입니다.
    {json.dumps(survey_json, ensure_ascii=False)}
    이 사용자의 투자 성향(위험 선호, 투자 기간, 목표, 정보처리 스타일 등)을 요약해줘.
    """
    client = openai.OpenAI()  # openai.api_key는 환경변수로 자동 인식
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"너는 투자 성향 분석 전문가야."},
            {"role":"user","content":prompt}
        ]
    )
    summary = response.choices[0].message.content
    return summary

if __name__ == "__main__":
    # (2) analysis_results.json 읽기
    with open('web/analysis_results.json', 'r', encoding='utf-8') as f:
        survey_json = json.load(f)

    # (3) LLM 분석
    summary = analyze_survey_with_llm(survey_json)

    # (4) supabase 저장
    supabase = get_supabase_client()
    data = {
        "user_id": "default",  # 실제 사용자 ID로 대체 가능
        "created_at": datetime.now().isoformat(),
        "profile_json": {
            "survey": survey_json,
            "summary": summary
        }
    }
    supabase.table("user_profiles").upsert(data).execute()
    print("supabase user_profiles 업서트 완료")
    print("분석 요약:", summary) 