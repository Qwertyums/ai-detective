import streamlit as st
import pandas as pd
from src.schemas import Document, Claim, Question
from src.retrieval import load_jsonl, build_context
from src.llm import run_llm_inference
from src.evaluation import evaluate_single_question, compare_modes

st.set_page_config(page_title="AI Detective", layout="wide")
st.title("🕵️ AI Detective: Оценка логического вывода LLM")

# ==========================================
# 1. Инициализация Session State (История)
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = {"A": [], "B": [], "C": []}
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None


# ==========================================
# 2. Загрузка данных
# ==========================================
@st.cache_data
def load_all_data():
    docs = load_jsonl("data/documents.jsonl", Document)
    claims = load_jsonl("data/claims.jsonl", Claim)
    questions = load_jsonl("data/questions.jsonl", Question)
    return docs, claims, questions


documents, claims, questions = load_all_data()

# ==========================================
# 3. Сайдбар: Конфигурация и Управление
# ==========================================
st.sidebar.header("Конфигурация")
mode = st.sidebar.radio(
    "Режим контекста",
    ["A: Полный контекст", "B: Плоский BM25 (Top-15)", "C: p-adic контекст"]
)

st.sidebar.header("Стресс-тесты")
permute_docs = st.sidebar.checkbox("Тест 1: Перестановка документов (seed=42)")
remove_premise = st.sidebar.checkbox("Тест 2: Удаление посылки (C25)")
add_contradiction = st.sidebar.checkbox("Тест 3: Добавление противоречия")

if st.sidebar.button("🗑️ Сбросить историю запусков"):
    st.session_state.history = {"A": [], "B": [], "C": []}
    st.session_state.last_result = None
    st.session_state.last_eval = None
    st.rerun()

# ==========================================
# 4. Основной экран: Выбор вопроса
# ==========================================
st.subheader("Выберите вопрос для анализа")
q_options = [f"{q.question_id}: {q.question}" for q in questions]
selected_q_str = st.selectbox("Вопрос", q_options)
selected_q = next(q for q in questions if selected_q_str.startswith(q.question_id))

# Подготовка параметров стресс-тестов
stress_remove = "C25" if remove_premise else None
stress_add = "C99: Админ признался, что сам запустил скрипт в 17:03." if add_contradiction else None

# Сборка контекста
mode_key = mode.split(":")[0].strip()  # "A", "B" или "C"
context = build_context(
    mode_key,
    selected_q.question,
    claims,
    documents,
    stress_remove_claim_id=stress_remove,
    stress_add_contradiction=stress_add
)

with st.expander("📄 Показать контекст, переданный модели"):
    st.text(context)

# ==========================================
# 5. Запуск инференса и Оценка
# ==========================================
if st.button("🚀 Запустить инференс", type="primary"):
    with st.spinner("Модель анализирует контекст..."):
        # 1. Получаем ответ от LLM
        result = run_llm_inference(selected_q, context)

        # 2. Оцениваем ответ против золотого стандарта
        eval_result = evaluate_single_question(selected_q, result)

        # 3. Сохраняем в историю
        st.session_state.last_result = result
        st.session_state.last_eval = eval_result
        st.session_state.history[mode_key].append(eval_result)

# ==========================================
# 6. Отображение результатов текущего запуска
# ==========================================
if st.session_state.last_result is not None:
    result = st.session_state.last_result
    eval_res = st.session_state.last_eval

    st.divider()
    st.subheader(f"Результат модели (Режим {mode_key})")

    # Метрики текущего запуска
    col1, col2, col3, col4 = st.columns(4)
    status_icon = "✅" if eval_res["status_accuracy"] == 1 else "❌"
    col1.metric("Статус", f"{status_icon} {result.status}")
    col2.metric("Confidence", f"{result.confidence:.2f}")
    col3.metric("Evidence F1", f"{eval_res['evidence_f1']:.2f}")
    col4.metric("Токены контекста", len(context.split()))

    # Детали ответа
    st.info(f"**Ответ модели:** {result.answer}")
    st.write(f"**Золотой ответ:** {selected_q.gold_answer}")

    st.write(f"**Объяснение:** {result.explanation}")

    col_evidence, col_counter = st.columns(2)
    with col_evidence:
        st.write(f"**Доказательства:** `{', '.join(result.evidence_ids) if result.evidence_ids else 'Нет'}`")
        st.write(f"**Золотые доказательства:** `{', '.join(selected_q.gold_evidence_ids)}`")

    with col_counter:
        st.write(
            f"**Контрдоказательства:** `{', '.join(result.counterevidence_ids) if result.counterevidence_ids else 'Нет'}`")
        st.write(f"**Золотые контрдоказательства:** `{', '.join(selected_q.gold_counterevidence_ids)}`")

# ==========================================
# 7. Сравнение режимов (Динамическая таблица)
# ==========================================
st.divider()
st.subheader("📊 Сравнение режимов (Накопленная статистика)")

# Проверяем, есть ли данные в истории
has_data = any(len(v) > 0 for v in st.session_state.history.values())

if has_data:
    df_comparison = compare_modes(st.session_state.history)

    # Переименуем колонки для красивого отображения в Streamlit
    df_comparison = df_comparison.rename(columns={
        "Mode": "Режим",
        "Status Accuracy": "Accuracy Статуса",
        "Evidence F1": "Evidence F1",
        "Counterevidence Recall": "Counter. Recall",
        "Both Recognition Rate": "Распознано 'both'",
        "Insufficient Recognition Rate": "Распознано 'insufficient'",
        "Mean Confidence": "Средний Confidence"
    })

    # Форматируем числа
    df_display = df_comparison.style.format({
        "Accuracy Статуса": "{:.2%}",
        "Evidence F1": "{:.2f}",
        "Counter. Recall": "{:.2f}",
        "Распознано 'both'": "{:.2%}",
        "Распознано 'insufficient'": "{:.2%}",
        "Средний Confidence": "{:.2f}"
    })

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.caption(
        "💡 Подсказка: Запустите инференс для одного и того же вопроса в режимах A, B и C, чтобы увидеть, как меняется Evidence F1 и распознавание сложных статусов.")
else:
    st.info("ℹ️ История пуста. Запустите инференс хотя бы в одном режиме, чтобы увидеть сравнительную аналитику.")
