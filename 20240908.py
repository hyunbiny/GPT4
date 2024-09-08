import json
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser

# JSON 파서 클래스
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()

# Streamlit 설정
st.set_page_config(page_title="QuizGPT", page_icon="❓")

st.title("QuizGPT")

# 사용자로부터 OpenAI API 키 입력받기
api_key = st.sidebar.text_input("Enter your OpenAI API Key", "")

# 난이도 선택
difficulty = st.sidebar.selectbox(
    "Select the difficulty of the questions",
    ("Easy", "Hard"),
)

# GitHub 레포지토리 링크 추가
st.sidebar.markdown("[GitHub Repo](https://github.com/hyunbiny/gpt4)")

# OpenAI API 키가 있을 때에만 실행
if api_key:
    # OpenAI 모델 초기화
    llm = ChatOpenAI(
        temperature=0.1 if difficulty == "Easy" else 0.5,
        model="gpt-4o-mini",
        api_key=api_key,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    # 질문 생성을 위한 프롬프트
    questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant that is role playing as a teacher.
                
                Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
                
                Each question should have 4 answers, three of them must be incorrect and one should be correct.
                
                Use (o) to signal the correct answer.
                
                Context: {context}
                """
            )
        ]
    )

    # 문서 형식화 함수
    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    # 퀴즈 체인 실행 함수
    def run_quiz_chain(docs):
        questions_chain = {"context": format_docs(docs)} | questions_prompt | llm | output_parser
        return questions_chain.invoke(docs)

    # 파일 업로드 시 문서 처리
    @st.cache_data(show_spinner="Loading file...")
    def split_file(file):
        file_content = file.read()
        file_path = f"./.cache/quiz_files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = TextLoader(file_path, encoding='utf8')
        docs = loader.load_and_split(text_splitter=splitter)
        return docs

    # Wikipedia 검색 처리
    @st.cache_data(show_spinner="Searching Wikipedia...")
    def wiki_search(term):
        retriever = WikipediaRetriever(top_k_results=5)
        docs = retriever.get_relevant_documents(term)
        return docs

    # 사이드바에서 문서 선택
    with st.sidebar:
        docs = None
        choice = st.selectbox(
            "Choose what you want to use.",
            ("File", "Wikipedia Article"),
        )
        if choice == "File":
            file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)

    # 문서가 선택되지 않은 경우
    if not docs:
        st.markdown(
            """
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
        )
    else:
        # 퀴즈 생성 및 출력
        response = run_quiz_chain(docs)
        correct_answers = 0
        total_questions = len(response["questions"])

        with st.form("questions_form"):
            for question in response["questions"]:
                st.write(question["question"])
                selected_answer = st.radio(
                    "Select an answer",
                    [answer["answer"] for answer in question["answers"]],
                    key=question["question"],
                )
                # 정답 확인
                for answer in question["answers"]:
                    if answer["answer"] == selected_answer and answer["correct"]:
                        correct_answers += 1

            # 퀴즈 제출 버튼
            submitted = st.form_submit_button("Submit")

        # 결과 표시
        if submitted:
            if correct_answers == total_questions:
                st.balloons()
                st.success("Congratulations! You got all answers correct!")
            else:
                st.error(f"You got {correct_answers} out of {total_questions} correct. Try again?")
                if st.button("Retry Quiz"):
                    st.experimental_rerun()

else:
    st.warning("Please enter your OpenAI API Key to start the quiz.")