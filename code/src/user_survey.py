# -*- coding: utf-8 -*-
import sys
import json
import uuid
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QMessageBox,
)
from PyQt5.QtCore import Qt
import requests
import time, os


class InvestmentAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.current_question_index = 0
        self.answers = []
        self.total_scores = {
            "risk_tolerance": 0,
            "investment_time_horizon": 0,
            "financial_goal_orientation": 0,
            "information_processing_style": 0,
            "investment_fear": 0,
            "investment_confidence": 0,
        }
        self.completion_executor = CompletionExecutor(
            host="https://clovastudio.stream.ntruss.com", 
            api_key="Bearer nv-e302186b2e7640d38c732700bd828020zBct",
            request_id='3d487f4d478e4a2a8a86d7d4fe509b76'
        )
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.question_label = QLabel()
        self.question_label.setWordWrap(True)
        self.question_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.question_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(self.question_label)

        self.answer_text = QTextEdit()
        layout.addWidget(self.answer_text)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_answer)
        layout.addWidget(self.submit_button)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setWindowTitle("Investment Analysis")
        self.setGeometry(300, 300, 1200, 700)
        self.setMinimumWidth(600)

        self.load_question()

    def load_question(self):
        if self.current_question_index < len(questions):
            self.question_label.setText(questions[self.current_question_index])
        else:
            self.question_label.setText("All questions answered. Analyzing results...")
            self.answer_text.setReadOnly(True)
            self.submit_button.setEnabled(False)
            self.analyze_results()

    def submit_answer(self):
        answer = self.answer_text.toPlainText()
        if answer.strip():
            self.answers.append(answer)
            self.process_answer(questions[self.current_question_index], answer)
            self.current_question_index += 1
            self.answer_text.clear()
            self.load_question()
        else:
            QMessageBox.warning(self, "Empty Answer", "Please provide an answer before submitting.")

    def process_answer(self, question, answer):
        # prompt_survey-score.txt 파일은 src 폴더 내에 있어야 합니다.
        prompt_template = load_prompt_template("prompt_survey-score.txt")
        if not prompt_template:
            QMessageBox.critical(self, "Error", "Failed to load prompt template.")
            return
        formatted_prompt = prompt_template.replace("[question]", question).replace("[answer]", answer)
        messages = [
            {"role": "system", "content": "너는 사용자의 요청을 엄격하게 json형태에 맞게 응답하는 AI야."},
            {"role": "user", "content": formatted_prompt},
        ]
        try:
            response = self.completion_executor.execute(messages)
            scores = json.loads(response)
            if scores:
                for key in self.total_scores:
                    self.total_scores[key] += scores.get(key, 0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing answer: {str(e)}")

    def analyze_results(self):
        # prompt_survey-analysis.txt 파일 이름이 정확한지 확인하세요.
        prompt_template = load_prompt_template("prompt_survey-analysis.txt")
        if not prompt_template:
            QMessageBox.critical(self, "Error", "Failed to load analysis prompt template.")
            return
        total_scores = self.total_scores
        formatted_prompt = (
            prompt_template.replace("[score1]", str(total_scores["risk_tolerance"]))
            .replace("[score2]", str(total_scores["investment_time_horizon"]))
            .replace("[score3]", str(total_scores["financial_goal_orientation"]))
            .replace("[score4]", str(total_scores["information_processing_style"]))
            .replace("[score5]", str(total_scores["investment_fear"]))
            .replace("[score6]", str(total_scores["investment_confidence"]))
        )
        messages = [
            {"role": "system", "content": "너는 사용자의 요청을 엄격하게 json형태에 맞게 응답하는 AI야."},
            {"role": "user", "content": formatted_prompt},
        ]
        try:
            response = self.completion_executor.execute(messages)
            # JSON 응답 정제
            response = response.strip()
            # 제어 문자 제거
            response = ''.join(char for char in response if ord(char) >= 32 or char in '\n\r\t')
            analysis = json.loads(response)
            print(analysis)
            self.save_analysis_results(analysis)
            result_text = "<h2>투자 성향 분석 결과</h2>"
            for key, value in analysis.items():
                if key != "overall_evaluation":
                    result_text += f"<h3>{key.replace('_analysis', '').replace('_', ' ').title()}</h3>"
                    result_text += f"<p>{value}</p>"
            result_text += "<h3>Overall Evaluation</h3>"
            result_text += f"<p>{analysis.get('overall_evaluation', 'Overall evaluation not available.')}</p>"
            self.result_label.setText(result_text)
            self.result_label.setTextFormat(Qt.RichText)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing results: {str(e)}")

    def save_analysis_results(self, analysis, filename="analysis_results.json"):
        try:
            # 현재 스크립트 파일의 디렉토리 경로를 가져옴
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # src 디렉토리에 파일 저장
            file_path = os.path.join(script_dir, filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=4)
            print(f"Analysis results saved to {file_path}")
            
            # 루트 디렉토리에도 동일한 파일 복사 (기존 코드와의 호환성을 위해)
            root_path = os.path.dirname(script_dir)
            root_file_path = os.path.join(root_path, filename)
            with open(root_file_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=4)
            print(f"Analysis results also copied to {root_file_path}")
        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")

    @staticmethod
    def load_analysis_results(filename="analysis_results.json"):
        # 현재 스크립트 파일의 디렉토리 경로를 가져옴
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # src 디렉토리에서 파일 경로 설정
        file_path = os.path.join(script_dir, filename)
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 루트 디렉토리에서도 확인
            root_path = os.path.dirname(script_dir)
            root_file_path = os.path.join(root_path, filename)
            if os.path.exists(root_file_path):
                with open(root_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                print(f"File {filename} not found in either src or root directory.")
                return None


class CompletionExecutor:
    def __init__(self, host, api_key, request_id, max_retries=3, retry_delay=60):
        self._host = host
        self._api_key = api_key  # API 키는 "Bearer ..." 형식이어야 합니다.
        self._request_id = request_id
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def execute(self, messages):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": self._api_key,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
        }
        data = {
            "messages": messages,
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 2048,
            "temperature": 0.1,
            "repeatPenalty": 1.1,
            "stopBefore": [],
            "includeAiFilters": False,
        }
        for attempt in range(self._max_retries):
            response = requests.post(
                f"{self._host}/testapp/v1/chat-completions/HCX-003",
                headers=headers,
                json=data,
            )
            result = response.json()
            if ("result" in result and result["result"] is not None and "message" in result["result"]):
                return result["result"]["message"]["content"]
            elif "status" in result and result["status"]["code"] == "42901":
                time.sleep(self._retry_delay)
            else:
                raise ValueError(f"Unexpected API response: {result}")
        raise ValueError(f"Failed to get a valid response after {self._max_retries} attempts.")


def load_prompt_template(prompt_file="prompt_survey-score.txt"):
    try:
        # 경로는 필요에 따라 수정하세요.
        prompt_path = os.path.join("c:/Users/work4/OneDrive/바탕 화면/투자챗봇/src", prompt_file)
        if not os.path.exists(prompt_path):
            print(f"Warning: Prompt file not found at {prompt_path}")
            return None
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"프롬프트 파일 로드 중 오류 발생: {e}")
        return None


def load_prompt2_template():
    try:
        # 현재 스크립트 파일의 디렉토리 경로를 가져옴
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(script_dir, "prompt_survey-analysis.txt")
        
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: prompt_survey-analysis.txt file not found in {script_dir}.")
        return None
    except IOError:
        print("Error: Could not read prompt_survey-analysis.txt file.")
        return None


questions = [
    "큰 꿈을 안고 시작한 투자, 여러분은 투자를 할 때 가장 중요한 요소가 무엇이라고 생각하나요? 안전하게 꾸준한 수익을 내는 투자가 옳은 투자일까요? 혹은 위험하더라도 큰 수익을 내야 진정한 투자일까요? 생각을 자유롭게 작성해 주세요.",
    "2024년 말 글로벌 경기침체 우려와 미국의 금리 인하 영향으로 증권 시장이 크게 요동쳤어요. 하지만 시간이 지나고 요동쳤던 증권 시장이 점차 안정화되었습니다. 증권 시장은 예상치 못한 이유로도 단기간에 변동되는 현상을 보여주기도 합니다. 여러분이라면 이런 단기적으로 크게 변동되는 상황을 어떻게 대응하시나요?",
    "투자는 제각각 다른 목표를 갖고 시작하곤 합니다. 누군가는 소소한 용돈을 벌기 위해, 누군가는 자가 구입을 위해 투자를 하고 있어요. 여러분의 투자 목표는 무엇인가요? 얼마나 수익을 내고, 언제까지 투자를 하고 싶나요?",
    "투자할 종목을 선택하려고 합니다. 어떤 정보를 참고하는 것이 좋을까요? 다양한 투자자들의 의견을 들어보기 위해 네이버 증권의 커뮤니티를 확인해 볼 수도 있고, 신뢰성있는 정보를 위해 뉴스나 전문가의 칼럼을 참고할 수도 있어요. 요즘 화두가 되는 AI 투자 분석 도구도 활용할 수 있습니다. 여러분들은 어떤 정보를 어떻게 활용할 예정인가요?",
    "high risk-high return 이라는 말을 들어보셨나요? 어떻게 생각하시나요? 특히 최근 암호화폐나 AI 관련 주식처럼 변동성이 큰 자산에 대해 어떻게 생각하시나요? 위험이 커도 높은 수익률을 추구하시나요? 아니면 수익이 낮더라도 안정적인 수익률이 나을까요?",
    "출근길 뉴스를 보니 내가 보유하고 있는 기술주와 관련된 안 좋은 소식이 보도되고 있어요. 그런데 간접적이기도 하고, 시장 반응도 꼭 나쁘지만은 않은 것 같아 보인다면, 어떻게 하실건가요? 주식을 매도하는게 나을까요? 혹은 기다리시나요? 선택과 이유를 함께 적어주세요.",
    "긴 고민 끝에 잘 만들어둔 나의 포트폴리오. 매일매일 변하는 평가손익이 자꾸 눈에 거슬리기도 합니다. 조언을 구해보면 일희일비 하지 말고, 앱을 지우는 것도 좋은 방법이라고 소개해 주었어요. 하지만 앱을 지운다면 포트폴리오를 자주, 그리고 즉각적으로 수정하기는 어려울 것 같아 고민입니다. 여러분은 앱을 지우고 목표 기간 뒤에 열어보는게 좋다고 생각하시나요? 혹은 매일매일 확인하고 포트폴리오를 즉각적으로 대응하여 수정하는 것이 좋다고 생각하시나요?",
    "우리는 지금까지 다양한 상황에서 판단을 해왔어요. 지금, 답변을 적는 이 순간 여러분의 투자 지식은 어느정도 된다고 생각하시나요? 최근의 경제 상황과 시장 변화에 대해 어느 정도 이해하고 계신가요? 자유롭게 기술해 주세요.",
    "나의 투자 실력을 곰곰이 생각하다보니 그 찰나에 나의 주식이 크게 떨어졌어요. 최근 AI 기술 기업들의 실적 부진과 관련이 있을 수도 있습니다. 지금 이 순간을 어떻게 대처하실건가요?",
    "재빠른 대처 능력으로 위기를 잘 극복해 내었습니다. 이젠 스스로도 투자를 어느정도 잘 하고 다음 과정으로 나아가도 되겠다는 생각이 들기도 합니다. 지금 이 생각을 한 순간 눈 앞에 지속가능 투자(ESG) 관련 새로운 금융 상품이나 신흥 시장 투자 기회가 있다면 어떤 기분과 생각이 드시나요?",
]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = InvestmentAnalysisApp()
    ex.show()
    sys.exit(app.exec_())
