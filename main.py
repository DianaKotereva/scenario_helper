import time

import streamlit as st
from app import app

# Настройка страницы
st.set_page_config(
    page_title="LangGraph Агент",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Инициализация состояния сессии
if "history" not in st.session_state:
    st.session_state.history = []


# Функция для получения ответа от агента
def get_agent_response(question):
    try:
        inputs = {"user_question": question}
        state_res = app.invoke(inputs)
        return state_res["final_answer"]
    except Exception as e:
        return f"Произошла ошибка: {str(e)}"


# Стилизация с помощью CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 1rem;
    }
    .history-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Боковая панель
with st.sidebar:
    st.title("О приложении")
    st.markdown("""
    Задай вопрос про Тень и Пламя, система подумает и выдаст ответ!
    """)

    # Очистка истории
    if st.button("Очистить историю", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Основной контент
st.markdown('<div class="main-header">🤖 LangGraph Агент</div>', unsafe_allow_html=True)

# Форма ввода вопроса
with st.form(key="question_form"):
    user_question = st.text_area(
        "💬 Ваш вопрос:", placeholder="Введите ваш вопрос здесь...", height=100
    )

    submit_button = st.form_submit_button("Получить ответ", use_container_width=True)

# Обработка вопроса
if submit_button and user_question:
    with st.spinner("🤔 Агент думает над ответом..."):
        start_time = time.time()
        response = get_agent_response(user_question)
        processing_time = time.time() - start_time

    # Сохранение в историю
    st.session_state.history.append(
        {
            "question": user_question,
            "answer": response,
            "timestamp": time.strftime("%H:%M:%S"),
        }
    )

    # Отображение ответа
    st.markdown("### 📝 Ответ:")
    st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)

    # Статистика
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"⏱️ Время обработки: {processing_time:.2f} сек")
    with col2:
        st.success(f"📊 Размер ответа: {len(response)} символов")

# История диалога
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📚 История диалога")

    for i, chat in enumerate(
        reversed(st.session_state.history[-5:])
    ):  # Последние 5 записей
        with st.expander(f"Вопрос: {chat['question'][:50]}... | {chat['timestamp']}"):
            st.markdown(f"**❓ Вопрос:** {chat['question']}")
            st.markdown(f"**🤖 Ответ:** {chat['answer']}")

# Информация в футере
st.markdown("---")
st.markdown(
    "*Приложение разработано для демонстрации агента аналитика книги с графовой структурой* • "
)
