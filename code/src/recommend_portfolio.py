import json
from datetime import datetime
from db_client import get_supabase_client

# 1. 사용자 성향 조회
supabase = get_supabase_client()
res = supabase.table("user_profiles").select("profile_json").eq("user_id", "default").execute()
if not res.data:
    raise ValueError("user_profiles에 해당 사용자가 없습니다.")
profile = res.data[0]["profile_json"]

# 2. 포트폴리오 추천 (LLM 활용)
def recommend_portfolio(profile_json):
    import openai
    prompt = f"""
    다음은 사용자의 투자 성향 분석 결과입니다.
    {json.dumps(profile_json, ensure_ascii=False)}
    이 투자자에게 적합한 국내주식 포트폴리오(종목 5~10개, 각 비중 포함)를 추천해줘.
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"너는 자산배분 전문가야."},
            {"role":"user","content":prompt}
        ]
    )
    recommendation = response.choices[0].message.content
    return recommendation

recommendation = recommend_portfolio(profile)

# 3. supabase에 추천 결과 저장
portfolio_data = {
    "user_id": "default",
    "created_at": datetime.now().isoformat(),
    "portfolio_json": {
        "profile": profile,
        "recommendation": recommendation
    }
}
supabase.table("portfolio_recommendations").upsert(portfolio_data).execute()
print("supabase portfolio_recommendations 업서트 완료")
print("추천 포트폴리오:\n", recommendation) 