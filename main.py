import torch
import google.generativeai as genai
from faster_whisper import WhisperModel
from pyannote.audio.pipelines.speaker_diarization import PretrainedSpeakerDiarizationPipeline
from pyannote.core import Segment
import os
import datetime
import re

# 🔹 [전역 변수] 설정 값 모아두기
class Config:
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # 🔑 Gemini API 키 (변경 필요)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부
    WHISPER_MODEL_SIZE = "medium"  # Whisper 모델 크기 선택 (small, medium, large 가능)
    SUPPORTED_AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac')  # 지원하는 음성 파일 확장자
    
    # 🔹 [전역 변수] LLM 프롬프트 템플릿
    SYSTEM_PROMPT = """
    당신은 회의록 작성 및 분석에 특화된 전문 비서입니다.
    주어진 회의록 원문과 화자 정보를 깊이 있게 분석하여, 
    회의 요약, 주요 결정 사항, 그리고 적절한 태그를 정확하게 생성하는 데 뛰어난 역량을 가지고 있습니다.
    회의록의 중요성을 인지하고, 핵심 내용만을 파악하는 능력을 가지고 있습니다.
    """

    USER_PROMPT_TEMPLATE = """
    주어진 회의록 원문과 각 발언의 화자 정보를 바탕으로, 다음 요구사항에 맞춰 회의록을 작성해 주세요.
    
    <요구 사항>
    1.  회의 요약: 회의에서 논의된 주요 내용을 3~5문장으로 간결하게 요약합니다.
    2.  주요 결정 사항: 회의에서 결정된 사항을 명확하게 목록 형태로 작성합니다.
        - 각 결정 사항은 간결하게 요약합니다.
    3.  태그: 회의 내용과 관련된 태그들을 해시태그(#) 형식으로 나열합니다. (#프로젝트, #의사결정, #TODO, #결정사항, #긴급, #참고사항 등). 태그는 3개 이상 작성합니다.
    
    <출력 형식>
    ## 회의 요약
    - [요약된 내용 1]
    - [요약된 내용 2]
    - ...

    ## 주요 결정 사항
    - [결정 사항 1]
    - [결정 사항 2]
    - ...

    ## 태그
    - #태그1
    - #태그2
    - ...

    <예시>
    ## 회의 요약
    - 이번 회의에서는 새로운 프로젝트 기획에 대한 논의가 진행되었습니다.
    - 마케팅 전략에 대한 구체적인 방향과 실행 계획을 수립하였습니다.
    - 다음 회의에서는 각 팀의 역할을 분담하고, 세부 추진 계획을 검토하기로 하였습니다.

    ## 주요 결정 사항
    - 신규 프로젝트 'Alpha'의 시작을 승인합니다.
    - 마케팅 팀은 다음 주까지 마케팅 전략에 대한 구체적인 계획을 제출해야 합니다.

    ## 태그
    - #프로젝트기획 #마케팅전략 #의사결정 #Alpha #TODO #긴급
    
    <회의록 원문>
    {speaker_transcript}
    """

    # 🔹 LLM(Gemini) 파라미터 설정 (새로 추가됨)
    TEMPERATURE = 0.4  # 0.0 ~ 1.0 사이의 값, 낮을수록 결정론적, 높을수록 창의적 (기본값: 0.9)
    TOP_P = 0.95  # 누적 확률 값, 높을수록 다양한 결과 생성 (기본값: 1.0)
    TOP_K = 40  # 확률이 높은 토큰 상위 K개만 고려 (기본값: 40)
    MAX_OUTPUT_TOKENS = 2048  # 최대 출력 토큰 수 (기본값: 2048)
    
# 설정 객체 생성
config = Config()

# 🔹 Whisper 모델 로드
whisper_model = WhisperModel(config.WHISPER_MODEL_SIZE, device=config.DEVICE, compute_type="float16")

# 🔹 Pyannote 화자 분리 모델 로드
diarization_pipeline = PretrainedSpeakerDiarizationPipeline.from_pretrained("pyannote/speaker-diarization")
diarization_pipeline.to(config.DEVICE)

# 🔹 Gemini API 설정
genai.configure(api_key=config.GEMINI_API_KEY)

def validate_audio_file(audio_path):
    """음성 파일의 유효성을 검사합니다."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ 오류: 파일을 찾을 수 없습니다. 올바른 경로를 입력하세요. ({audio_path})")
    if not audio_path.lower().endswith(config.SUPPORTED_AUDIO_EXTENSIONS):
        raise ValueError(f"❌ 오류: 지원하지 않는 파일 형식입니다. ({audio_path}) 지원 형식: {config.SUPPORTED_AUDIO_EXTENSIONS}")

def transcribe_audio(audio_path):
    """Whisper를 사용하여 음성을 텍스트로 변환합니다."""
    print("🔹 [1/4] 음성 파일 변환 시작...")
    try:
        segments, _ = whisper_model.transcribe(audio_path, word_timestamps=True)
        transcript = [f"{segment.start:.2f}-{segment.end:.2f}: {segment.text.strip()}" for segment in segments]
        print("✅ Whisper 변환 완료!")
        return transcript
    except Exception as e:
        raise RuntimeError(f"❌ 오류: Whisper 변환 중 오류가 발생했습니다: {e}")

def identify_speakers(audio_path):
    """Pyannote를 사용하여 화자를 식별합니다."""
    print("🔹 [2/4] 화자 분리 수행 중...")
    try:
        diarization_result = diarization_pipeline(audio_path)
        print("✅ 화자 분리 완료!")
        return diarization_result
    except Exception as e:
        raise RuntimeError(f"❌ 오류: 화자 분리 중 오류가 발생했습니다: {e}")

def process_speaker_segments(diarization_result, transcript):
    """화자 분리 결과를 바탕으로 발언을 그룹화합니다."""
    speaker_segments = {}
    
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_segments.setdefault(speaker, []).append(turn)
    
    speaker_transcript = []
    for speaker, segments in speaker_segments.items():
        speaker_text = ""
        for segment in segments:
          for transcript_text in transcript:
            start,end,text = re.split(r'[-:]',transcript_text)
            if float(start) <= segment.start and float(end) >= segment.end:
              speaker_text += f'{text} '
              
        if len(speaker_text) > 0:
          speaker_transcript.append(f"{speaker}: {speaker_text.strip()}")

    return speaker_transcript

def summarize_meeting(speaker_transcript):
    """Gemini를 사용하여 회의록을 요약합니다."""
    print("🔹 [3/4] LLM을 사용한 요약 생성 중...")
    try:
        # System 메시지와 User 메시지를 분리하여 전달
        prompt = [
            {"role": "system", "parts": [config.SYSTEM_PROMPT]},
            {"role": "user", "parts": [config.USER_PROMPT_TEMPLATE.format(speaker_transcript="\n".join(speaker_transcript))]},
        ]
        
        # LLM 파라미터를 config에서 가져와서 설정 (변경됨)
        response = genai.generate_text(
            contents=prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                max_output_tokens=config.MAX_OUTPUT_TOKENS
            )
        )
        meeting_summary = response.text
        print("✅ 요약 생성 완료!")
        return meeting_summary
    except Exception as e:
        raise RuntimeError(f"❌ 오류: 요약 생성 중 오류가 발생했습니다: {e}")

def save_meeting_log(meeting_summary, speaker_transcript):
    """회의록을 Markdown 파일로 저장합니다."""
    print("🔹 [4/4] Markdown 저장 중...")
    try:
        filename = f"meeting_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# 📝 회의록\n\n")
            f.write(f"## 📌 요약\n{meeting_summary}\n\n")
            f.write(f"## 🔊 원문\n")
            for line in speaker_transcript:
                f.write(f"- {line}\n")
            f.write(f"\n_(이 회의록은 AI에 의해 자동 생성되었습니다)_\n")

        print(f"✅ 회의록 저장 완료! 파일: {filename}")
    except Exception as e:
        raise RuntimeError(f"❌ 오류: 파일 저장 중 오류가 발생했습니다: {e}")

def generate_meeting_summary(audio_path):
    """전체 회의록 생성 프로세스를 관리합니다."""
    try:
        validate_audio_file(audio_path)
        print(f"📂 입력된 파일: {audio_path}")

        transcript = transcribe_audio(audio_path)
        diarization_result = identify_speakers(audio_path)
        speaker_transcript = process_speaker_segments(diarization_result, transcript)
        meeting_summary = summarize_meeting(speaker_transcript)
        save_meeting_log(meeting_summary, speaker_transcript)
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(e)
    finally:
        # 메모리 해제 (필요에 따라 추가)
        del transcript
        del diarization_result
        del speaker_transcript
        del meeting_summary

# 🔹 실행: 사용자 입력으로 파일 경로 받기
if __name__ == "__main__":
    audio_path = input("🎤 음성 파일 경로 입력: ").strip()
    generate_meeting_summary(audio_path)
