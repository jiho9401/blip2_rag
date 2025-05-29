import os
import json
import traceback
from dotenv import load_dotenv
from autorag.deploy import Runner

# .env 파일 로드
load_dotenv()

# API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter your OpenAI API Key: ").strip()
    os.environ["OPENAI_API_KEY"] = api_key

    # .env에 기존 키가 있는지 확인 후 저장
    if "OPENAI_API_KEY" not in open(".env", "r", encoding="utf-8").read():
        with open(".env", "a", encoding="utf-8") as env_file:
            env_file.write(f"\nOPENAI_API_KEY={api_key}\n")

print("✅ OpenAI API Key has been set successfully!")

def generate_query(blip2_output, question):
    """
    blip2_output를 기반으로 question에 대한 적절한 RAG 쿼리를 생성.
    """
    return f"Based on the given text: '{blip2_output.strip()}', answer the question: '{question.strip()}'."


def process_dataset(json_path, output_path, trial_folder):
    """
    JSON 데이터를 읽고 RAG 기반 쿼리를 실행한 후 결과를 저장.
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {json_path}")
        return

    if not os.path.exists(trial_folder):
        print(f"[ERROR] trial_folder 경로를 찾을 수 없습니다: {trial_folder}")
        return

    # Runner 초기화
    try:
        runner = Runner.from_trial_folder(trial_folder)
        print(f"[INFO] RAG Runner Initialized with trial folder: {trial_folder}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Runner: {e}")
        return

    # JSON 파일 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파일 로드 실패: {e}")
        return

    # 처리된 항목 수
    processed_count = 0

    for item in data:
        if "rag_output" in item and item["rag_output"]:
            continue  # 이미 처리된 데이터는 건너뛰기

        blip2_output = item.get("blip2_output", "").strip()
        question = item.get("question", "").strip()

        if blip2_output and question:
            query_for_rag = generate_query(blip2_output, question)
            print(f"[INFO] Processing Query: {query_for_rag}")

            try:
                rag_result = runner.run(query_for_rag)
                item["rag_output"] = rag_result.strip() if rag_result else "EMPTY"
            except Exception as e:
                print(f"[ERROR] Failed to process query for question: {question}")
                print(traceback.format_exc())
                item["rag_output"] = "ERROR"

            processed_count += 1

    # 결과 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"[INFO] Updated dataset saved to {output_path}")
        print(f"[INFO] 총 {processed_count}개의 항목이 처리되었습니다.")
    except Exception as e:
        print(f"[ERROR] JSON 저장 중 오류 발생: {e}")


# 실행 파라미터
json_input_path = "vqa_test_dataset_300_anon_updated.json"
json_output_path = "vqa_test_dataset_300_anon_updated_rag.json"
trial_folder_path = "./benchmark/9"  # 실제 경로 확인 필요

# 데이터 처리 실행
process_dataset(json_input_path, json_output_path, trial_folder_path)
