# server.py (本地端電腦的 Web API) - 最終邏輯修復版

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import shutil
from pathlib import Path
import json
import uuid
import numpy as np
import traceback

from model_loader import ModelManager
from services import AIServices
import config
import utils

# -------------------------------------------------------------
# 伺服器初始化
# -------------------------------------------------------------
app = Flask(__name__)
CORS(app)
print("🤖 正在初始化伺服器端 AI 模型...")

models = ModelManager()
services = AIServices(models)

session_states = {}
TTS_OUTPUT_DIR = Path(config.TEMP_DIR) / "tts_outputs"
TTS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# -------------------------------------------------------------
# 核心處理 API
# -------------------------------------------------------------
@app.route(config.API_PROCESS_ENDPOINT, methods=['POST'])
def process_multimodal():
    # 1. 檔案接收與暫存
    if 'audio_file' not in request.files or 'turn_data' not in request.form:
        return jsonify({"error": "Missing audio file or turn data"}), 400

    temp_dir_name = str(uuid.uuid4())
    temp_dir = Path(config.TEMP_DIR) / temp_dir_name
    temp_dir.mkdir(exist_ok=True, parents=True)

    audio_file = request.files['audio_file']

    try:
        turn_data = json.loads(request.form['turn_data'])
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid turn_data format"}), 400

    session_id = turn_data.get('session_id', 'default_session')

    temp_audio_path = temp_dir / f"uploaded_{audio_file.filename}"
    audio_file.save(temp_audio_path)

    image_paths = []
    for key, file in request.files.items():
        if key.startswith('image_'):
            temp_image_path = temp_dir / file.filename
            file.save(temp_image_path)
            image_paths.append(str(temp_image_path))

    # 2. 狀態管理
    if session_id not in session_states:
        session_states[session_id] = {
            'history': [],
            'memory': utils.EmotionMemory(),
            'manual_personality': None
        }
    state = session_states[session_id]

    # 3. 核心 AI 運算
    try:
        # 3.1 STT (語音轉文字)
        user_audio_np, final_audio_path = utils.load_wav_to_numpy(str(temp_audio_path), config.SAMPLE_RATE)

        if user_audio_np is None:
            raise Exception("無法處理上傳的音訊，音訊可能損壞或為空。")

        user_text = services.transcribe(user_audio_np)
        utils.debug_log(f"STT 結果: {user_text}", "INFO")

        # 3.3 指令檢查 (切換人格)
        language = utils.detect_language(user_text)
        is_command, target = utils.check_personality_switch_command(user_text, language)

        new_manual_personality = None
        reply = None
        # 預設變數
        selected_personality = state['manual_personality'] or config.DEFAULT_PERSONALITY
        calc_logs = []

        if is_command:
            if target == 'reset':
                state['manual_personality'] = None
                new_manual_personality = None
                reply = "好的，我已將人格模式切換回自動偵測。"
            else:
                state['manual_personality'] = target
                new_manual_personality = target
                selected_personality = target
                reply = f"好的，我已將我的個性切換為【{config.PERSONALITY_CONFIGS[target]['name']}】模式。"

            tts_filename = f"tts_{uuid.uuid4()}.wav"
            tts_path = TTS_OUTPUT_DIR / tts_filename
            services.text_to_speech_cosvoice(reply, tts_path, selected_personality)
            shutil.rmtree(temp_dir)

            # 指令模式直接回傳
            return jsonify({
                "reply": reply,
                "tts_audio_file": tts_filename,
                "user_text": user_text,
                "personality": selected_personality,
                "updated_history": state['history'],
                "new_manual_personality": new_manual_personality,
                "is_wake_detected": True,
                "emotion_results": None,
                "calculation_logs": ["指令模式：不進行情緒運算"]
            })

        # 3.4 情緒分析 (取得原始資料)
        text_emotion = services.detect_text_emotion(user_text, language)
        audio_emotion = services.detect_audio_emotion(final_audio_path, language)
        facial_emotion = services.analyze_facial_emotion_from_images(image_paths)

        # 🟢 步驟 A: 先計算分數，找出贏家 (Winner)
        winner_emotion = "neutral"  # 預設

        if not state['manual_personality']:
            # 呼叫 services.py (注意：現在回傳 3 個值)
            selected_personality, calc_logs, winner_emotion = services.select_personality_auto(
                text_emotion['emotion'] if text_emotion else None,
                audio_emotion['emotion'] if audio_emotion else None,
                facial_emotion['emotion'] if facial_emotion else None,
                state['memory'],
                text_emotion['score'] if text_emotion else 0,
                audio_emotion['score'] if audio_emotion else 0,
                facial_emotion['confidence'] if facial_emotion else 0
            )

        # 🟢 步驟 B: 將「贏家」存入記憶
        # 這裡傳入 winner，讓記憶系統知道誰贏了
        state['memory'].add_emotion(
            text_emotion['emotion'] if text_emotion else None,
            audio_emotion['emotion'] if audio_emotion else None,
            facial_emotion['emotion'] if facial_emotion else None,
            winner=winner_emotion
        )

        # 🟢 步驟 C: 現在記憶更新了，檢查趨勢是否惡化
        if not state['manual_personality']:
            trend = state['memory'].get_emotion_trend()
            worsening_count = state['memory'].get_worsening_count()  # 獲取連續次數

            # [修改] 檢查連續負面回合數是否 >= 2 (代表連續兩輪或更多)
            if worsening_count >= 2:
                utils.debug_log(f"🚨 連續負面情緒 {worsening_count} 次，強制切換至【安撫型】", "WARNING")
                selected_personality = "comforting"
                calc_logs.append(f"🚨 記憶趨勢偵測: 連續負面 {worsening_count} 次 -> 強制切換安撫型")
            elif trend in ["improving", "persistent_improving"]:
                # 如果沒有連續負面，但趨勢是改善，仍記錄改善趨勢
                calc_logs.append(f"👍 記憶趨勢偵測: 情緒改善 ({trend})")
            elif trend in ["worsening", "persistent_worsening"]:
                # 趨勢雖惡化，但未達連續兩次的嚴格標準，僅記錄不切換
                calc_logs.append(f"🚨 記憶趨勢偵測: {trend} (未達連續負面兩次條件)")

        # LLM 生成回應
        reply, updated_history = services.generate_response(
            user_text, selected_personality, state['history'],
            text_emotion, audio_emotion, facial_emotion
        )
        state['history'] = updated_history

        # TTS 生成
        tts_filename = f"tts_{uuid.uuid4()}.wav"
        tts_path = TTS_OUTPUT_DIR / tts_filename
        services.text_to_speech_cosvoice(reply, tts_path, selected_personality)

        # 4. 清理暫存檔案
        shutil.rmtree(temp_dir)

        # 5. 回傳結果 (這裡之前可能遺失了)
        emotion_trend = state['memory'].get_emotion_trend()

        return jsonify({
            "reply": reply,
            "tts_audio_file": tts_filename,
            "user_text": user_text,
            "personality": selected_personality,
            "updated_history": state['history'],
            "new_manual_personality": state['manual_personality'],
            "is_wake_detected": True,
            "emotion_trend": emotion_trend,
            "calculation_logs": calc_logs,
            "emotion_results": {
                "text": text_emotion, "audio": audio_emotion, "facial": facial_emotion
            }
        })

    except Exception as e:
        utils.debug_log(f"處理 API 請求失敗: {e}", "ERROR")
        traceback.print_exc()

        if 'temp_dir' in locals() and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

        return jsonify({"error": f"Internal Server Error: {e}"}), 500


# -------------------------------------------------------------
# TTS 音訊下載 API
# -------------------------------------------------------------
@app.route(config.API_AUDIO_DOWNLOAD_ENDPOINT, methods=['GET'])
def download_tts_audio():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "Missing filename parameter"}), 400

    audio_path = TTS_OUTPUT_DIR / filename

    if audio_path.exists() and audio_path.is_file():
        return send_file(audio_path, mimetype="audio/wav", as_attachment=False)
    else:
        return jsonify({"error": f"File not found: {filename}"}), 404


# -------------------------------------------------------------
# 伺服器啟動
# -------------------------------------------------------------
if __name__ == '__main__':
    utils.debug_log("✨ 所有模型與裝置初始化完成！", "SUCCESS")

    try:
        port = config.FLASK_PORT
    except AttributeError:
        port = 8000

    utils.debug_log(f"✨ Flask 伺服器啟動中，請用 Web Tunnel 連線至 http://0.0.0.0:{port} ...", "INFO")

    app.run(host='0.0.0.0', port=port, debug=False)