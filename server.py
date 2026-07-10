# -*- coding: utf-8 -*-
import os
import requests
import json
import numpy as np
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask_cors import CORS

def fetch_real_news(coin_name):
    query = f"{coin_name} 코인"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        titles = []
        for item in root.findall('.//item')[:3]:
            title = item.find('title')
            if title is not None and title.text:
                titles.append(title.text.split(' - ')[0].strip())
        return titles
    except Exception as e:
        print(f"[*] fetch_real_news error: {str(e)}")
        return []

# NVIDIA cuDF Pandas 가속 활성화 시도
try:
    import cudf.pandas
    cudf.pandas.install()
    print("[+] GPU Acceleration enabled successfully for server backend.")
except ImportError:
    print("[-] GPU cuDF not found. Running server on default CPU Pandas.")

import pandas as pd

app = Flask(__name__)
CORS(app)  # 모든 오리진에 대한 CORS 허용

OLLAMA_API_URL = "http://localhost:11434/api/chat"

class TechnicalIndicators:
    @staticmethod
    def calculate_wilder_rsi(df, period=14):
        close_delta = df['close'].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        
        ma_up = up.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        ma_down = down.ewm(com=period - 1, adjust=False, min_periods=period).mean()
        
        rs = ma_up / (ma_down + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

class RealTimeBacktester:
    def __init__(self, initial_balance=10000000, fee_rate=0.0005, slippage=0.001):
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage

    def run(self, rsi_period=14, rsi_buy_threshold=30, rsi_sell_threshold=70):
        # 가상 퀀트 국면 데이터셋 생성 (cuDF 가속 구동)
        np.random.seed(42)
        base_price = 50000.0
        prices = [base_price]
        for i in range(1, 400):
            change = np.random.normal(0.0005, 0.008)
            prices.append(prices[-1] * (1 + change))
            
        df = pd.DataFrame({'close': prices})
        df['rsi'] = TechnicalIndicators.calculate_wilder_rsi(df, rsi_period)
        
        balance = self.initial_balance
        position = 0.0
        trades_count = 0
        entry_price = 0.0
        
        for i in range(len(df)):
            rsi = df.loc[i, 'rsi']
            close = df.loc[i, 'close']
            if pd.isna(rsi):
                continue
                
            # 매수 시그널 (RSI 과매도 돌파)
            if position == 0.0 and rsi < rsi_buy_threshold:
                buy_price = close * (1 + self.slippage)
                position = (balance * (1 - self.fee_rate)) / buy_price
                balance = 0.0
                entry_price = buy_price
                trades_count += 1
            # 매도 시그널 (RSI 과매수 돌파)
            elif position > 0.0 and rsi > rsi_sell_threshold:
                sell_price = close * (1 - self.slippage)
                balance = position * sell_price * (1 - self.fee_rate)
                position = 0.0
                trades_count += 1

        final_portfolio = balance if position == 0.0 else (position * df.iloc[-1]['close'] * (1 - self.fee_rate))
        total_return = ((final_portfolio - self.initial_balance) / self.initial_balance) * 100
        
        # MDD 계산
        equity_curve = []
        bal = self.initial_balance
        pos = 0.0
        for i in range(len(df)):
            rsi = df.loc[i, 'rsi']
            close = df.loc[i, 'close']
            if pos > 0.0:
                equity_curve.append(pos * close)
            else:
                equity_curve.append(bal)
                
            if pos == 0.0 and rsi < rsi_buy_threshold:
                pos = bal / close
                bal = 0.0
            elif pos > 0.0 and rsi > rsi_sell_threshold:
                bal = pos * close
                pos = 0.0
                
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd
                
        return {
            'total_return': round(total_return, 2),
            'mdd': round(max_dd, 2),
            'trades_count': trades_count
        }

@app.route('/api/backtest', methods=['POST'])
def handle_backtest():
    data = request.get_json() or {}
    rsi_period = int(data.get('rsi_period', 14))
    rsi_buy = int(data.get('rsi_buy', 30))
    rsi_sell = int(data.get('rsi_sell', 70))
    
    backtester = RealTimeBacktester()
    results = backtester.run(rsi_period, rsi_buy, rsi_sell)
    return jsonify(results)

@app.route('/api/ai-opinion', methods=['POST'])
def handle_ai_opinion():
    data = request.get_json() or {}
    coin_name = data.get('name', '비트코인')
    symbol = data.get('symbol', 'BTC')
    stars = data.get('stars', 0)
    commits = data.get('commits', 0)
    change_rate = data.get('change_rate', 0.0)
    recent_commits = data.get('recent_commits', [])
    latest_release = data.get('latest_release', None)
    desc = data.get('description', '')
    
    # 실시간 구글 뉴스 검색 크롤링 연동
    real_news = fetch_real_news(coin_name)
    news_text = ""
    if real_news:
        news_text = "\n".join([f"- 실제 뉴스 헤드라인: {title}" for title in real_news])
    else:
        news_text = "최근 24시간 내 시장 뉴스 보도나 새로운 호재/악재 소식이 검색되지 않았습니다."
        
    # 팩트 이력 텍스트화
    commits_text = ""
    if recent_commits and len(recent_commits) > 0:
        commits_text = "\n".join([f"- 커밋 내용: {c.get('message', '')} (날짜: {c.get('date', '')})" for c in recent_commits])
    else:
        commits_text = "최근 4주 동안 기록된 실제 코드 커밋 로그가 없거나 조회되지 않았습니다."
        
    release_text = "배포 이력 정보 없음"
    if latest_release:
        release_text = f"최신 배포 버전 태그: {latest_release.get('tag_name', '')} (배포일: {latest_release.get('published_at', '')})"
    
    # 밈코인 또는 데이터 미수집 Fallback 처리 분기점
    if stars == 0 and commits == 0:
        prompt = f"""
        [가상자산 실제 마켓 및 프로젝트 팩트 정보]
        이름: {coin_name} ({symbol})
        GitHub Stars: {stars}개 (오픈소스 활동이 부재하거나 비공개 프로젝트 상태)
        최근 24시간 가격 변동률: {change_rate}%
        공식 소개문구: {desc}
        최근 실제 커밋 로그: {commits_text}
        실제 최신 배포 릴리스 버전: {release_text}
        실시간 최신 시장 뉴스 헤드라인:
        {news_text}

        위 가상자산에 대해 수집된 실시간 최신 뉴스 헤드라인 및 시장 요소를 퀀트 관점에서 스캔하여 최근 상황을 사실에 기반하여 요약해 주십시오.
        반드시 다음 세 가지 항목을 명확하게 단락을 나누어 한글 존댓말로 전문성 있게 설명해 주십시오 (줄바꿈을 넣어 가독성 높게 작성):
        
        1. 🧠 로컬 AI 종합 의견: 해당 가상자산의 최근 실제 프로젝트 상태와 최신 뉴스를 통한 진단 요약
        2. 📈 주요 호재(Bullish) 및 새로운 소식: 수집된 실제 뉴스 헤드라인 기사 내용에 기반한 프로젝트의 긍정적 모멘텀 요약
        3. 📉 주요 악재(Bearish) 및 리스크: 오픈소스 코드 개발이 부재한 점, 또는 수집된 뉴스나 가격 변동성 상의 위험 요인 경고
        
        [🚨 중요 구속사항]
        - 절대로 거짓 소식, 모의 데이터, 허위 릴리스, 상상의 파트너십 같은 '거짓 정보(환각)'를 작성하지 마십시오.
        - 반드시 제공된 최근 뉴스 헤드라인 문구를 분석 내용 내에 문자 그대로 직접 인용(예: '최근 뉴스에 따르면~' 등)하여 호재와 악재의 신뢰성을 확보하십시오. 팩트 데이터 이외의 일반론적인 진단은 억제하십시오.
        - 정보가 없는 경우 억지로 지어내지 말고 '최근 뉴스 없음' 또는 '실제 개발 이력 확인 불가'라고 명확히 작성하십시오.
        """
    else:
        prompt = f"""
        [가상자산 실제 마켓 및 프로젝트 팩트 정보]
        이름: {coin_name} ({symbol})
        GitHub Stars: {stars}개
        최근 4주 GitHub Commits: {commits}회
        최근 24시간 가격 변동률: {change_rate}%
        공식 소개문구: {desc}
        최근 실제 커밋 로그: {commits_text}
        실제 최신 배포 릴리스 버전: {release_text}
        실시간 최신 시장 뉴스 헤드라인:
        {news_text}

        위 실제 깃허브 소스 코드 커밋 활성도, 릴리스 배포 기록, 그리고 실시간 최신 뉴스 헤드라인을 기반으로 기술적 안전성, 미래 가치, 그리고 최신 시장 소식(호재/악재)을 정밀 분석해 주십시오.
        반드시 다음 세 가지 항목을 명확하게 단락을 나누어 한글 존댓말로 전문성 있게 설명해 주십시오 (줄바꿈을 넣어 가독성 높게 작성):
        
        1. 🧠 로컬 AI 종합 의견: 실제 소스코드 개발 통계 및 배포 상태, 최신 시장 뉴스를 종합한 퀀트 진단
        2. 📈 주요 호재(Bullish) 및 새로운 소식: 실제 커밋 메시지 내용, 릴리스 버전 이력, 또는 실시간 뉴스 기사 내용에 기반한 구체적인 개발 진척과 시장의 호재 요약
        3. 📉 주요 악재(Bearish) 및 리스크: 실제 코드 배포 지연 여부, 실시간 뉴스상의 악재 소식, 최근 가격 변동에 따른 잠재적 리스크 요인
        
        [🚨 중요 구속사항]
        - 절대로 거짓 소식, 모의 데이터, 허위 릴리스, 상상의 파트너십 같은 '거짓 정보(환각)'를 작성하지 마십시오.
        - 반드시 제공된 실제 최근 커밋 메시지 내용(예: 커밋 내용 일부 문구)이나 최신 릴리스 버전명(예: v31.1 등), 혹은 수집된 실시간 뉴스 헤드라인 기사 제목을 분석 내용 내에 문자 그대로 최소 1개 이상 직접 인용하여 분석의 신뢰성을 확보하십시오. 팩트 데이터 이외의 일반론적인 진단은 억제하십시오.
        - 정보가 없거나 모호한 경우 '실제 개발 이력 확인 불가'라고 명확히 작성하십시오.
        """

    # NVIDIA NIM API 연동 가속 활성화 분기
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "")
    if nvidia_api_key:
        nim_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Content-Type": "application/json"
        }
        nim_payload = {
            "model": "meta/llama3-70b-instruct",
            "messages": [
                {"role": "system", "content": "당신은 가상자산의 개발 지표와 최신 뉴스, 소식을 스캔하여 오직 제공된 실제 데이터 팩트에 기반하여 위험/호재 요소를 발췌하는 정밀 퀀트 분석가 AI 에이전트입니다. 절대 근거 없는 거짓(환각) 소식을 쓰면 안 됩니다."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False
        }
        try:
            # NVIDIA NIM API 클라우드 초고속 응답 시도 (대략 1~2초 내 응답)
            response = requests.post(nim_url, headers=headers, json=nim_payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                opinion = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                return jsonify({'opinion': opinion})
            else:
                print(f"[*] NVIDIA NIM API returned error {response.status_code}. Falling back to Local Ollama...")
        except Exception as e:
            print(f"[*] NVIDIA NIM API exception: {str(e)}. Falling back to Local Ollama...")

    # 로컬 Ollama (Hermes-3) 디폴트 연동 처리
    payload = {
        "model": "hermes3",
        "messages": [
            {"role": "system", "content": "당신은 가상자산의 개발 지표와 최신 뉴스, 소식을 스캔하여 오직 제공된 실제 데이터 팩트에 기반하여 위험/호재 요소를 발췌하는 정밀 퀀트 분석가 AI 에이전트입니다. 절대 근거 없는 거짓(환각) 소식을 쓰면 안 됩니다."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=45)
        if response.status_code == 200:
            result = response.json()
            opinion = result.get('message', {}).get('content', '').strip()
            return jsonify({'opinion': opinion})
        else:
            return jsonify({'opinion': f"로컬 AI 엔진이 11434 포트에서 응답하였으나 오류를 회신했습니다 (HTTP {response.status_code})."})
    except requests.exceptions.RequestException as e:
        return jsonify({'opinion': f"로컬 AI 서버 통신 지연 또는 미기동 상태입니다. (상태 로그: {str(e)})"})

@app.route('/api/translate-profile', methods=['POST'])
def handle_translate_profile():
    data = request.get_json() or {}
    description = data.get('description', '')
    commits = data.get('commits', [])
    
    if not description and not commits:
        return jsonify({'description': '', 'commits': []})
        
    payload = {
        "description": description,
        "commits": commits
    }
    
    prompt = f"""
    Translate the following English description and commit messages into natural, professional, developer-friendly Korean.
    You must return a raw JSON object and nothing else. Do not wrap the JSON in markdown blocks (such as ```json) or write any other text.

    Input JSON data:
    {json.dumps(payload, ensure_ascii=False)}

    Output JSON Format:
    {{
        "description": "Translated Korean description",
        "commits": [
            "Translated Korean commit message 1",
            "Translated Korean commit message 2",
            ...
        ]
    }}
    """
    
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "")
    opinion = ""
    if nvidia_api_key:
        nim_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Content-Type": "application/json"
        }
        nim_payload = {
            "model": "meta/llama3-70b-instruct",
            "messages": [
                {"role": "system", "content": "You are a professional software translator. Translate English IT, blockchain, and Git commit messages into natural Korean. Output ONLY valid raw JSON matching the requested structure."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": False
        }
        try:
            response = requests.post(nim_url, headers=headers, json=nim_payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                opinion = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            else:
                print(f"[*] NVIDIA NIM API translation returned error {response.status_code}. Falling back to Local Ollama...")
        except Exception as e:
            print(f"[*] NVIDIA NIM API translation exception: {str(e)}. Falling back to Local Ollama...")

    if not opinion:
        ollama_payload = {
            "model": "hermes3",
            "messages": [
                {"role": "system", "content": "You are a professional software translator. Translate English IT, blockchain, and Git commit messages into natural Korean. Output ONLY valid raw JSON matching the requested structure."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        try:
            response = requests.post(OLLAMA_API_URL, json=ollama_payload, timeout=45)
            if response.status_code == 200:
                result = response.json()
                opinion = result.get('message', {}).get('content', '').strip()
            else:
                print(f"[*] Local Ollama translation returned error {response.status_code}.")
        except Exception as e:
            print(f"[*] Local Ollama translation exception: {str(e)}")

    translated_desc = description
    translated_commits = commits
    if opinion:
        try:
            if "```" in opinion:
                parts = opinion.split("```")
                # json 마크다운 펜스가 포함된 경우
                for p in parts:
                    p_str = p.strip()
                    if p_str.startswith("json"):
                        p_str = p_str[4:].strip()
                    if p_str.startswith("{") and p_str.endswith("}"):
                        opinion = p_str
                        break
            parsed = json.loads(opinion.strip())
            translated_desc = parsed.get('description', description)
            translated_commits = parsed.get('commits', commits)
        except Exception as e:
            print(f"[*] Failed to parse translation JSON: {str(e)}, raw: {opinion}")
            
    return jsonify({
        'description': translated_desc,
        'commits': translated_commits
    })

@app.route('/api/anomaly-check', methods=['GET'])
def handle_anomaly():
    # 업비트 주요 가상자산 Ticker 스캔하여 이상 지표 확인
    url = "https://api.upbit.com/v1/ticker"
    params = {"markets": "KRW-BTC,KRW-ETH,KRW-XRP,KRW-DOGE,KRW-SOL"}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            tickers = response.json()
            anomalies = []
            for t in tickers:
                market = t.get('market', '')
                change_rate = t.get('signed_change_rate', 0.0) * 100
                acc_trade_price_24h = t.get('acc_trade_price_24h', 0.0)
                
                # 급락/급등 이상 징후 감지 (절대값 3.0% 이상 변동) 또는 수급 폭발 (24시간 누적 거래대금 2천억 원 초과)
                if abs(change_rate) >= 3.0:
                    anomalies.append(f"{market.replace('KRW-', '')} 24H 변동성 급증 감지 ({round(change_rate, 2)}%)")
                elif acc_trade_price_24h >= 200000000000:
                    anomalies.append(f"{market.replace('KRW-', '')} 거래대금 수급 폭발 감지 (약 {round(acc_trade_price_24h / 100000000, 1)}억)")
            
            if anomalies:
                return jsonify({'status': 'ANOMALY', 'message': " | ".join(anomalies)})
            else:
                return jsonify({'status': 'NORMAL', 'message': '가상자산 원화 마켓의 단기 변동성 및 수급 상태가 안정적 범위에 있습니다.'})
        else:
            return jsonify({'status': 'UNKNOWN', 'message': '업비트 Ticker 데이터를 수신하지 못했습니다.'})
    except Exception as e:
        return jsonify({'status': 'UNKNOWN', 'message': f'이상 탐지 체크 오류: {str(e)}'})

if __name__ == '__main__':
    # 5000 포트에서 백그라운드 웹 서비스 오픈
    app.run(host='0.0.0.0', port=5000)
