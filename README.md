# MeetingNoteGenerator

**AI 기반 자동 회의록 생성기**

## 📖 프로젝트 개요

MeetingNoteGenerator는 음성 파일을 입력받아, AI 기술을 활용하여 자동으로 회의록을 생성하는 프로그램입니다. Whisper를 사용하여 음성을 텍스트로 변환하고, Pyannote를 통해 화자를 분리하며, Gemini API를 활용하여 회의 내용을 요약하고 주요 결정 사항 및 태그를 생성합니다. 최종적으로, 회의록은 Markdown 파일로 저장됩니다.

## 🎯 주요 기능

*   **음성 텍스트 변환 (Whisper):** 다양한 음성 파일을 텍스트로 변환합니다.
*   **화자 분리 (Pyannote):** 각 발언이 누구의 것인지 식별합니다.
*   **AI 기반 요약 (Gemini):** 회의 내용을 자동으로 요약하고, 주요 결정 사항을 추출합니다.
*   **태그 생성:** 회의 내용에 맞는 적절한 태그를 자동으로 생성합니다.
*   **Markdown 저장:** 요약, 원문, 태그, 시간 정보 등을 포함한 회의록을 Markdown 파일로 저장합니다.
* **LLM 파라미터 조절**: 다양한 LLM 파라미터 조절을 통해, 원하는 품질의 회의록을 만들 수 있습니다.
* **system & user prompt**: 시스템, 유저 프롬프트를 분리하여, Gemini에게 더 명확하게 지시합니다.

## 🛠️ 기술 스택

*   **Python:** 프로그램 개발 언어
*   **Faster-Whisper:** 음성 인식 라이브러리
*   **Pyannote Audio:** 화자 분리 라이브러리
*   **Google Gemini API:** 텍스트 요약 및 태그 생성에 활용되는 LLM API
*   **PyTorch:** 딥러닝 프레임워크 (Whisper, Pyannote 기반)
* **notion-client**: 노션 API 연동 (추후 업데이트)

## ⚙️ 설치 및 사용법

1.  **필수 라이브러리 설치:**
    ```bash
    pip install torch google-generativeai faster-whisper pyannote.audio
    ```

2.  **API 키 설정:**
    *   Google Gemini API 키를 발급받고, `main.py` 파일의 `Config` 클래스 내 `GEMINI_API_KEY` 변수에 입력합니다.
        ```python
        class Config:
          GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
        ```
    *  (추후 업데이트) 노션 API 키가 있다면, `NOTION_API_KEY` 변수에 입력합니다.

3.  **실행:**
    ```bash
    python main.py
    ```
    실행 후, 음성 파일 경로를 입력하라는 메시지가 나오면 경로를 입력합니다.

4. **LLM 파라미터 변경**
    * 필요에 따라, `Config` 클래스의 `TEMPERATURE`, `TOP_P`, `TOP_K`, `MAX_OUTPUT_TOKENS`를 변경할 수 있습니다.
    ```python
    TEMPERATURE = 0.4
    TOP_P = 0.95
    TOP_K = 40
    MAX_OUTPUT_TOKENS = 2048
    ```
    * `TEMPERATURE`: 생성되는 텍스트의 다양성을 조절합니다.
        * `0.0`: 가장 결정론적인 결과를 생성합니다. 같은 입력에 대해 항상 동일한 출력이 생성됩니다.
        * `1.0`: 가장 창의적이고 다양한 결과를 생성합니다.
    * `TOP_P`: 누적 확률을 고려하여 토큰을 선택합니다.
        * `1.0`: 모든 토큰을 고려합니다.
        * `0.9`: 누적 확률이 90%가 되는 지점까지만 토큰을 고려합니다. 나머지 10%의 확률을 가진 토큰들은 무시됩니다.
    * `TOP_K`: 상위 K개의 토큰만 고려합니다.
        * `40`: 확률이 높은 상위 40개의 토큰만 고려합니다.
    * `MAX_OUTPUT_TOKENS`: 생성되는 텍스트의 최대 길이를 토큰 단위로 제한합니다.

## 📁 파일 구성

*   `main.py`: 메인 소스 코드 파일
* `requirements.txt` : 필요한 라이브러리 정보

## 📝  향후 개선 사항 (To-Do)

* **노션 연동:** 노션 API를 사용하여 자동으로 회의록을 노션에 저장하는 기능 추가
* **더욱 강화된 예외 처리:** 다양한 예외 상황에 대한 대처 기능 추가
* **UI 개선:** 사용자가 좀 더 편리하게 사용할 수 있도록 UI 개선
* **다양한 LLM 지원:** gemini 외에도 다른 llm 모델을 선택할 수 있도록 개선

## 🤝 기여하기

프로젝트 개선에 기여하고 싶으신 분들은 언제든지 Pull Request를 보내주세요!

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.
