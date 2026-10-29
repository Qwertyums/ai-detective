"""Build a short, honest report directly from persisted experimental records."""
from html import escape
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_validation import ROOT, load_dataset, validate_dataset
from src.experiments import load_records
from src.llm import configuration
from scripts.summarize_results import summarize


def number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def build():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    docs, claims, questions = load_dataset()
    size = validate_dataset(docs, claims, questions)
    records = load_records()
    report_model, report_settings = configuration(include_profile=False)
    result = summarize(records)
    (ROOT/'results/summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    baseline_rows = [row for row in result['metrics'] if row['kind'] == 'baseline' and row['provider'] == report_settings['provider']]
    observations = 'Завершённых сопоставимых троек пока нет; эмпирическое сравнение режимов не установлено.'
    if baseline_rows:
        series = max({row['study_digest'] for row in baseline_rows}, key=lambda key: sum(row['attempts'] for row in baseline_rows if row['study_digest'] == key))
        source_record = next(r for r in records if r['study_digest'] == series)
        report_model, report_settings = source_record['model'], source_record['settings']
        selected = {row['mode']: row for row in baseline_rows if row['study_digest'] == series}
        observations = 'Основная сопоставимая серия ' + series[:8] + '. '
        observations += 'Accuracy по всем попыткам (включая ошибки): ' + '; '.join(mode+' '+number(row.get('accuracy_including_errors')) for mode,row in selected.items()) + '. '
        observations += 'Распознавание both среди валидных ответов своего класса: ' + '; '.join(mode+' '+number(row.get('Both Recognition Rate')) for mode,row in selected.items()) + '. '
        observations += 'Распознавание insufficient: ' + '; '.join(mode+' '+number(row.get('Insufficient Recognition Rate')) for mode,row in selected.items()) + '. '
        observations += 'Доля уверенных утверждений при недостатке данных: ' + '; '.join(mode+' '+number(row.get('Confident Unsupported Rate')) for mode,row in selected.items()) + '. '
        observations += 'Средняя уверенность и фактическая accuracy валидных ответов: ' + '; '.join(mode+' '+number(row.get('Mean Confidence'))+'/'+number(row.get('Status Accuracy')) for mode,row in selected.items()) + '. Отсутствующая оценка означает отсутствие валидных ответов нужного класса, а не нулевой результат. '
        from src.experiments import paired_records
        groups = {}
        for record in paired_records(records):
            if record['study_digest'] == series and record['kind'] == 'baseline':
                groups.setdefault(record['question_id'], {})[record['mode']] = record
        helped, harmed = [], []
        for key, group in groups.items():
            b, c = group['B'].get('evaluation'), group['C'].get('evaluation')
            if b and c and b['status_accuracy'] != c['status_accuracy']:
                (helped if c['status_accuracy'] > b['status_accuracy'] else harmed).append(key)
        observations += 'C улучшил статус относительно B на вопросах: '+(', '.join(helped) or 'нет наблюдений')+'; ухудшил: '+(', '.join(harmed) or 'нет наблюдений')+'. Для выводов об устойчивости нужны проверенные стресс-прогоны.'
    connection = 'подписку ChatGPT с OAuth' if report_settings['provider'] == 'chatgpt' else 'отдельный API-биллинг'
    transport = ('store=false и stream=true; max_output_tokens и truncation не передаются'
                 if report_settings['provider'] == 'chatgpt' else
                 f'store=false, truncation=disabled и максимум {report_settings.get("max_output_tokens", 2048)} выходных токенов')
    sections = [
        ('Цель и материал', f'Исследуется поиск и организация доказательств в полностью вымышленном деле о сбое на турнире. Досье содержит {size["documents"]} документов, {size["words"]} слов, {size["claims"]} атомарных утверждений и {size["questions"]} вопросов: по два прямых, multi-hop, временных, с правилом и исключением, конфликтом и недостатком данных. D08 и D10 — тематически похожие отвлекающие документы. Истинная рабочая история отделена от контекста модели. Противоречия в названии команды и свидетельствах намеренно не разрешены.'),
        ('Метод', f'Используется OpenAI Responses API через {connection}, модель {report_model}. A получает полные документы с индексом claim_id. B использует BM25 top-15. C получает те же записи, сгруппированные по p-adic ветвям и отсортированные внутри по времени события. Промпт и настройки едины: строгая JSON-схема, {transport}, reasoning effort {report_settings["reasoning_effort"]}. Температура и seed не передаются. Исторические попытки локальной Qwen, API-ключа и подписки сохраняются раздельными сериями. Наличие кода не подтверждает завершение эксперимента.'),
        ('Организация доказательств', 'При p=3 путь состоит из типа источника, логического статуса и времени. Код равен a0 + 3a1 + 9a2; расстояние определяется числом общих начальных уровней. Источник поставлен первым, чтобы отделять машинные записи, свидетельства и нормы до временной сортировки. Если поставить время первым, соседними станут разнородные сообщения одного периода; это может помочь временным вопросам, но разнести конфликтующие свидетельства. Само дерево не устанавливает истинность и не разрешает конфликт автоматически.'),
        ('Стрессовые вмешательства', 'Подготовлены перестановки для 12 вопросов с seed=42, удаления одного основания для всех 12 вопросов и добавления независимого несовместимого источника для 12 вопросов. Всего 36 сценариев, по три режима — 108 стресс-запросов; вместе с базой — 144. При удалении одной стороны конфликта Q9/Q10 ожидается entailed; у Q11/Q12 сохраняется insufficient. Эти эталоны требуют ручной проверки, в том числе редакторских пересказов удалённой позиции. Удаление затрагивает исходную цитату и аннотацию. Вопросам о неизвестной личности и векторе атаки добавление конфликта в посылке не даёт недостающих данных: insufficient может сохраниться. Сценарии без реальной ручной проверки не оцениваются.'),
        ('Состояние эксперимента', f'Сохранено {len(records)} фактических попыток, в полные сопоставимые тройки вошло {result["paired_records"]}. Ошибок загрузки, генерации или формата: {sum(r["error"] is not None for r in records)}. Стресс-попыток без подтверждённой ручной проверки: {sum(r["kind"] != "baseline" and not r["reviewed"] for r in records)}. Каждый JSON содержит сырой ответ, точный контекст, состав до/после, порядок документов, параметры, промпт, ревизию модели и измерения. Ошибки не выдаются за логический отказ.'),
        ('Метрики', 'Accuracy и evidence precision/recall/F1 считаются для валидных ответов, ошибки показываются отдельно и дополнительно учитываются как неуспех в accuracy по всем попыткам. Counterevidence recall считается только для эталонов с контрдоказательствами. Распознавание both и insufficient оценивается внутри соответствующего класса. Для недостатка информации фиксируется доля содержательных ответов с confidence ≥0,8. Устойчивость включает смену статуса и ответа, Jaccard evidence_ids и соответствие новому эталону. Токены берутся из API usage, время измеряется вокруг сетевого запроса. При повторном запросе затраты суммируются.'),
        ('Наблюдаемые результаты', observations),
        ('Ответы на исследовательские вопросы', 'Сравнение retrieval с полным текстом и flat с p-adic следует проводить по одной и той же серии и одинаковым вопросам. Улучшение распознавания конфликтов проверяется отдельно на both. Отличение недостатка данных от догадки — на insufficient и уверенных неподтверждённых утверждениях. Перестановка должна сохранять основания; удаление проверяет зависимость от необходимой посылки; добавление конфликта — готовность сохранять обе позиции. Калибровка confidence сопоставляет заявленную уверенность с фактической accuracy по диапазонам. Польза или вред p-adic структуры устанавливаются по конкретным расхождениям ответов и цитат, а не по виду дерева.'),
        ('Ограничения и вывод', 'Базовые эталоны требуют совместной проверки участниками. Без проверенных сценариев стресс-метрики не являются завершённым экспериментом. Один сюжет и два вопроса каждого типа ограничивают обобщение. Индекс утверждений увеличивает объём A и облегчает ссылки, поэтому режим A следует описывать с этим условием. Формальная валидация не доказывает верность свободного объяснения: требуется ручная проверка. Если ниже нет оценённых результатов, эмпирические выводы о преимуществе retrieval, p-adic группировки или калибровке confidence пока не установлены. Реальный вклад людей должен быть указан ими в CONTRIBUTIONS.md.'),
        ('Источники и воспроизведение', 'Условие: «Лабораторные работы АОТ.docx», лабораторная № 6. Structured Outputs: https://developers.openai.com/api/docs/guides/structured-outputs . Данные, промпт, тесты, команды запуска и правила ручной проверки находятся в репозитории; точные результаты — в results/runs, сводка — в results/summary.json.'),
    ]
    markdown = ['# AI Detective исследование доказательств', '']
    for title, text in sections:
        markdown += ['## '+title, '', text, '']
    rows = result['metrics']
    if rows:
        markdown += ['## Результаты сопоставимых троек', '', '| Серия | Условие | Режим | Попыток | Оценено | Ошибок | Accuracy | Evidence F1 |', '|---|---|---|---:|---:|---:|---:|---:|']
        for row in rows:
            markdown.append(f'| {row["study_digest"][:8]} | {row["kind"]} | {row["mode"]} | {row["attempts"]} | {row["valid_scored"]} | {row["errors"]} | {number(row.get("Status Accuracy"))} | {number(row.get("Evidence F1"))} |')
    (ROOT/'report.md').write_text('\n'.join(markdown), encoding='utf-8')
    if len(' '.join(markdown).split()) > 1500:
        raise ValueError('Report exceeds 1500 words')
    font_root = Path('C:/Windows/Fonts')
    pdfmetrics.registerFont(TTFont('Body', str(font_root/'arial.ttf')))
    pdfmetrics.registerFont(TTFont('BodyBold', str(font_root/'arialbd.ttf')))
    body = ParagraphStyle('Body', fontName='Body', fontSize=10.5, leading=15, spaceAfter=7, alignment=TA_LEFT)
    heading = ParagraphStyle('Heading', parent=body, fontName='BodyBold', fontSize=12, leading=16, spaceBefore=9, spaceAfter=5, keepWithNext=True)
    title_style = ParagraphStyle('Title', parent=heading, fontSize=18, leading=23, spaceAfter=14)
    story = [Paragraph('AI Detective<br/>Исследование доказательств', title_style)]
    for title, text in sections:
        story += [Paragraph(escape(title), heading), Paragraph(escape(text), body)]
    if rows:
        story.append(Paragraph('Результаты сопоставимых троек', heading))
        headers = ['Условие', 'Режим', 'Попыток', 'Оценено', 'Ошибок', 'Accuracy', 'Ev. F1']
        cells = [[Paragraph(escape(x), body) for x in headers]]
        for row in rows:
            cells.append([Paragraph(escape(str(x)), body) for x in [row['kind'], row['mode'],row['attempts'],row['valid_scored'],row['errors'], number(row.get('Status Accuracy')), number(row.get('Evidence F1'))]])
        table = Table(cells, colWidths=[34*mm,19*mm,23*mm,23*mm,23*mm,23*mm,21*mm], repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
        story.append(table)
    def page_number(canvas, doc):
        canvas.setFont('Body',9)
        canvas.drawRightString(190*mm,12*mm,str(doc.page))
    SimpleDocTemplate(str(ROOT/'report.pdf'), pagesize=(210*mm,297*mm), leftMargin=22*mm, rightMargin=22*mm,
                      topMargin=18*mm,bottomMargin=19*mm,title='AI Detective исследование доказательств',author='Учебный проект').build(story,onFirstPage=page_number,onLaterPages=page_number)
    print('Report words:', len(' '.join(markdown).split()))


if __name__ == '__main__':
    build()
