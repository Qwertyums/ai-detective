import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.schemas import ModelOutput, Question
from pydantic import ValidationError

# Глобальные переменные для кэширования модели (чтобы не грузить её при каждом клике)
_model = None
_tokenizer = None


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    global _model, _tokenizer
    if _model is None:
        print(f"⏳ Загрузка модели {model_name}... (это может занять время)")
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Используем float16 для экономии памяти. Если у вас GPU с поддержкой bfloat16, можно заменить.
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        _model.eval()
        print("✅ Модель успешно загружена в память.")


def run_llm_inference(question: Question, context: str) -> ModelOutput:
    """
    Выполняет инференс на Qwen2.5-7B и возвращает валидированный ModelOutput.
    """
    load_model()

    system_prompt = "Ты — строгий аналитик данных. Твоя задача — анализировать контекст и отвечать ИСКЛЮЧИТЕЛЬНО в формате валидного JSON. Никакого текста до или после JSON."

    user_prompt = f"""Контекст:
{context}

Вопрос: {question.question}

Проанализируй контекст и верни ответ СТРОГО в следующем JSON-формате (без markdown-оберток, без лишнего текста):
{{
  "status": "entailed | contradicted | both | insufficient",
  "answer": "краткий ответ или null",
  "evidence_ids": ["C01", "C02"],
  "counterevidence_ids": ["C05"],
  "confidence": 0.95,
  "explanation": "Краткое объяснение связи между доказательствами"
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Применяем chat template для Qwen
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)

    # Генерация
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,  # Низкая температура для строгого соблюдения формата
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    # Декодируем только сгенерированную часть
    response_text = _tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Попытка извлечь и распарсить JSON
    try:
        # Очистка от возможных markdown-оберток, если модель их всё-таки добавила
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(response_text)

        # Добавляем ID вопроса, так как модель его не возвращает
        parsed["question_id"] = question.question_id

        # Приведение типов (модель может вернуть confidence как строку)
        if "confidence" in parsed:
            parsed["confidence"] = float(parsed["confidence"])
        else:
            parsed["confidence"] = 0.5

        # Гарантируем наличие списков
        parsed.setdefault("evidence_ids", [])
        parsed.setdefault("counterevidence_ids", [])

        # Валидация через Pydantic (проверит статусы, диапазоны и правила both/insufficient)
        return ModelOutput(**parsed)

    except (json.JSONDecodeError, ValidationError, KeyError, ValueError) as e:
        print(f"❌ Ошибка парсинга/валидации ответа LLM: {e}")
        print(f"Сырой ответ модели:\n{response_text[:500]}...")

        # Fallback: возвращаем безопасный ответ с нулевой уверенностью
        return ModelOutput(
            question_id=question.question_id,
            status="insufficient",
            answer="Ошибка генерации или парсинга ответа модели.",
            evidence_ids=[],
            counterevidence_ids=[],
            confidence=0.0,
            explanation=f"Модель вернула невалидный JSON. Сырой ответ: {response_text[:150]}..."
        )