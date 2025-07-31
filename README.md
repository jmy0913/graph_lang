## 1 dotenv 설정

- LANGSMITH_API_KEY=
- LANGSMITH_ENDPOINT=https://api.smith.langchain.com/
- skn14_langchain=
- LANGSMITH_TRACING=
- DART_API_KEY=
- OPENAI_API_KEY=
- NAVER_CLIENT_ID=
- NAVER_CLIENT_SECRET=


## 2. 벡터 db 설정 : utils2/ 폴더 안에 각 벡터 db 2개를 폴더 자체로 넣어놔야 함.
- 예시 : util2/faiss_index3(안에 파일 2개), util2/faiss_index_bge_m3(안에 파일 2개)
- 다운 링크 : https://drive.google.com/drive/folders/19y5kH1-mgCo3-0_Rbuxq3gCFL7zoI9ar?usp=drive_link


## 3. requirements.txt 다운로드 필요
- 터미널 => 가상환경 활성화 => pip install
- 명령어 : pip install -r requirements.txt


## 4. 코드 파일
- 기존과 동일한 부분 :
  - api_get.py
  - normalize_code_search.py
- 변경된 부분 : 
  - retriever_setting.py : faiss 벡터 db retriever로 수정됨
  - chain_setting.py : chat_history를 최근 대화목록으로 주도록 prompt 수정됨.
  - graph_node.py : 랭그래프 각 노드에 사용될 함수들 정의 (각 chain 실행 함수들)
  - graph_setting.py : 각 노드 함수들을 조합해서 만든 랭그래프 
  - main.py : 랭그래프를 불러와서, 실행시키는 최종 실행 함수 (함수인자: 질문,세션id,레벨,채팅기록)


## 5. 그래프 실행 방법
- utils2/main.py에 있는 run_graph 함수 import해서 사용
  - 함수 인자 : user_input, config_id, level='basic', chat_history=None
  - user_input : 사용자 질문
  - level : 'basic', 'intermediate', 'advanced' 중에 설정
  - chat_history : 나중에 채팅 기록 db에서 불러와서 셋팅해주는 용도(안쓰면 빈 값으로 시작)
  - config_id : 세션 id
  - 예시 (아래 코드)
```python 
from utils2.main import run_langraph

run_langraph('2024년 삼성전자 재무제표 알려줘', 'id_1', level='basic')
```
