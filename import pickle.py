import pickle

file_path = r"C:\Users\work4\OneDrive\바탕 화면\투자챗봇\data\processed\financial_statements.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("✅ 데이터 로딩 완료")