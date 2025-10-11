import gradio as gr
from app import app


def answer_question(question):
    """Функция для обработки вопросов через LangGraph агента"""
    try:
        inputs = {"user_question": question}
        state_res = app.invoke(inputs)
        return state_res["final_answer"]
    except Exception as e:
        return f"Произошла ошибка при обработке запроса: {str(e)}"


# Создаем интерфейс с помощью Blocks для большей кастомизации
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="LangGraph AI Агент",
    css="""
    .container { max-width: 800px; margin: auto; padding: 20px; }
    .header { text-align: center; margin-bottom: 30px; }
    """,
) as demo:
    # Заголовок и описание
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            # 🤖 LangGraph AI Агент
            **Интеллектуальная система для ответов на сложные вопросы**
            
            Задайте любой вопрос и получите развернутый, структурированный ответ от AI агента.
            """,
            elem_classes="header",
        )

        # Основная область ввода/вывода
        with gr.Row():
            with gr.Column(scale=1):
                input_question = gr.Textbox(
                    label="💬 Ваш вопрос",
                    placeholder="Введите ваш вопрос здесь...",
                    lines=4,
                    max_lines=8,
                    elem_id="question_input",
                )

                submit_btn = gr.Button(
                    "🚀 Получить ответ", variant="primary", size="lg"
                )

                clear_btn = gr.Button("🧹 Очистить", variant="secondary")

            with gr.Column(scale=1):
                output_answer = gr.Textbox(
                    label="📝 Ответ агента",
                    placeholder="Здесь появится ответ на ваш вопрос...",
                    lines=8,
                    max_lines=12,
                    show_copy_button=True,
                    elem_id="answer_output",
                )

        # Примеры вопросов
        with gr.Accordion("📋 Примеры вопросов для тестирования", open=False):
            gr.Examples(
                examples=[
                    [
                        "В чем принципиальное различие между Чистыми и Искателями совершенства? Структура, методы, идеология и т.д."
                    ],
                    ["Объясните основные концепции искусственного интеллекта"],
                    [
                        "Каковы преимущества и недостатки разных подходов к машинному обучению?"
                    ],
                    ["Как современные AI системы обрабатывают естественный язык?"],
                ],
                inputs=input_question,
                label="Нажмите на пример чтобы автоматически заполнить поле ввода",
            )

        # Информация о системе
        with gr.Accordion("ℹ️ Информация о системе", open=False):
            gr.Markdown(
                """
                ### Технические детали:
                - **Backend**: LangGraph Agent
                - **Интерфейс**: Gradio
                - **Обработка**: Семантический анализ и генерация ответов
                - **Обновления**: Реальное время
                
                ### Рекомендации по использованию:
                - Формулируйте вопросы четко и конкретно
                - Используйте примеры выше для тестирования
                - Для сложных тем задавайте уточняющие вопросы
                """
            )

    # Обработчики событий
    submit_btn.click(
        fn=answer_question,
        inputs=input_question,
        outputs=output_answer,
        api_name="answer",
    )

    clear_btn.click(
        fn=lambda: ("", ""), inputs=None, outputs=[input_question, output_answer]
    )


# Функция запуска
def main():
    demo.launch(
        server_name="0.0.0.0",  # Доступ с других устройств в сети
        server_port=7860,
        share=False,  # Установите True для создания публичной ссылки
        inbrowser=True,  # Автоматически открывать браузер
        show_error=True,
        favicon_path=None,
        quiet=False,
    )


if __name__ == "__main__":
    main()
