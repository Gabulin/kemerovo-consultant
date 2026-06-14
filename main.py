"""
RAG-консультант по льготам для многодетных семей г. Кемерово
Backend: FastAPI | Python 3.11 | Groq API (Llama 3.3 70B)

Версия 3.0 — переработка после аудита:
- профиль закреплён в system-контексте с приоритетом над историей
- единый источник истины: краткий реестр мер генерируется, не дублируется вручную
- temperature снижена до 0.2 для фактологической точности
- двухуровневая проверка релевантности (порог + тематический якорь)
- промпт ориентирован на консультативный диалог, а не выдачу списков
- логирование запросов для оценки качества (метрики для диссертации)
"""

import os, re, math, json, datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Основной провайдер — Groq (Llama 3.3 70B): быстрый, щедрые лимиты, без обрыва ответов
# Резервный провайдер — Google Gemini (google-genai SDK, при 429/ошибке Groq)
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

app = FastAPI()

# ═══════════════════════════════════════════
#  RAG: загрузка и индексация
# ═══════════════════════════════════════════

KB_FILES = [
    "kb/01_federal.md", "kb/02_regional.md", "kb/03_special.md",
    "kb/04_organizations.md", "kb/05_documents.md",
]

def load_and_chunk(filepath, chunk_size=800, overlap=150):
    try:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"[WARN] Не найден: {filepath}")
        return []
    sections = re.split(r'\n(?=#{1,3} )', text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec or len(sec) < 30:
            continue
        if len(sec) <= chunk_size:
            chunks.append(sec)
        else:
            words = sec.split()
            step = max(1, (chunk_size - overlap) // 6)
            wsize = chunk_size // 6
            for i in range(0, len(words), step):
                part = " ".join(words[i:i + wsize])
                if len(part) > 50:
                    chunks.append(part)
    return chunks

def tok(text):
    return re.findall(r'[а-яёa-z0-9]+', text.lower())

def build_index(chunks):
    N = len(chunks)
    df = {}
    tf_list = []
    for ch in chunks:
        tokens = tok(ch)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = sum(tf.values()) or 1
        tf = {t: v / total for t, v in tf.items()}
        tf_list.append(tf)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(N / (v + 1)) + 1 for t, v in df.items()}
    vecs = [{t: tf[t] * idf.get(t, 0) for t in tf} for tf in tf_list]
    return idf, vecs

def cos(a, b):
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

# Тематический якорь: запрос должен пересекаться с лексикой домена.
# Решает проблему «рецепт борща» (score 0.33 из-за слова «рецепт»).
DOMAIN_ANCHORS = {
    "льгот", "выплат", "пособи", "компенсац", "субсиди", "капитал", "ипотек",
    "многодетн", "семь", "ребен", "ребён", "дет", "детск", "школ", "сад", "садик",
    "удостоверен", "налог", "проезд", "питани", "лекарств", "земл", "участок",
    "соцзащит", "мфц", "сфр", "госуслуг", "документ", "оформ", "получ", "положен",
    "право", "статус", "форм", "пенси", "вуз", "колледж", "обучен", "жку", "коммунал",
    "квартир", "помощь", "поддержк", "деньг", "рубл", "сумм", "куда", "адрес", "телефон",
    "контракт", "юрист", "юридическ", "извещател", "газ", "молочн", "кухн",
}

def is_on_topic(query):
    """Проверяет, относится ли запрос к домену льгот (тематический якорь)."""
    ql = query.lower()
    return any(anchor in ql for anchor in DOMAIN_ANCHORS)

def retrieve(query, top_k=6, min_score=0.05):
    # Расширение запроса: для организационных вопросов добавляем ключи
    q_expanded = query
    org_triggers = ("адрес", "телефон", "режим", "мфц", "сфр", "куда идти",
                    "куда обратиться", "куда обращаться", "где находится", "где получить")
    if any(t in query.lower() for t in org_triggers):
        q_expanded = query + " адрес телефон режим работы Кемерово отдел"

    qt = tok(q_expanded)
    qtf = {}
    for t in qt:
        qtf[t] = qtf.get(t, 0) + 1
    total = sum(qtf.values()) or 1
    qvec = {t: (qtf[t] / total) * IDF.get(t, 1) for t in qtf}
    scores = sorted(enumerate(VECS), key=lambda x: cos(qvec, x[1]), reverse=True)

    # Мета-паттерны которые не несут пользы пользователю
    META_PATTERNS = (
        "ВЫВОДЫ ДЛЯ RAG", "АЛГОРИТМЫ ОТВЕТОВ", "RAG База знаний",
        "ЭТАП 1", "ЭТАП 2", "ЭТАП 3", "ЧАСТЬ А.", "ЧАСТЬ Б.", "ЧАСТЬ В.",
        "ЧАСТЬ Г.", "ЧАСТЬ Д.", "ЧАСТЬ Е.", "ЧАСТЬ Ж.", "ЧАСТЬ З.",
        "методическое замечание", "Актуальность: июнь 2026",
        "Нейроконсультант vs", "АНАЛИЗ УДОБСТВА",
    )

    results, seen = [], set()
    for idx, _ in scores:
        sc = cos(qvec, VECS[idx])
        if sc < min_score:
            break
        chunk = CHUNKS[idx]

        # 1. Отсекаем слишком короткие чанки (заголовки без содержания)
        if len(chunk) < 80:
            continue

        # 2. Отсекаем мета-чанки (написаны для системы, не для пользователя)
        if any(p in chunk for p in META_PATTERNS):
            continue

        # 3. Дедупликация
        key = chunk[:100]
        if key not in seen:
            seen.add(key)
            results.append(chunk)
        if len(results) >= top_k:
            break
    return results

print("[RAG] Загрузка базы знаний...")
CHUNKS = []
for f in KB_FILES:
    c = load_and_chunk(f)
    CHUNKS.extend(c)
    print(f"  {f}: {len(c)} чанков")
print(f"[RAG] Итого: {len(CHUNKS)} чанков")
IDF, VECS = build_index(CHUNKS)
print(f"[RAG] Индекс: {len(IDF)} токенов")

# ── Контакты отделов соцвыплат по районам (парсим из базы, единый источник) ──
# Чтобы бот всегда давал ПРАВИЛЬНЫЙ адрес/телефон отдела района пользователя
# и не путал его с МФЦ — подкладываем нужный блок в контекст детерминированно.
DISTRICT_KEYS = ("центральн", "ленинск", "заводск", "рудничн", "кировск")

def _load_district_offices():
    offices = {}
    try:
        text = open("kb/04_organizations.md", encoding="utf-8").read()
    except FileNotFoundError:
        return offices
    for m in re.finditer(r'### А\.\d[^\n]*\n.*?(?=\n### |\n## |\Z)', text, re.S):
        block = m.group(0).strip()
        header = block.split("\n", 1)[0].lower()
        for kw in DISTRICT_KEYS:
            if kw in header:
                offices[kw] = block
                break
    return offices

DISTRICT_OFFICES = _load_district_offices()
print(f"[RAG] Отделы соцвыплат по районам: {len(DISTRICT_OFFICES)}")

def district_contact_block(profile):
    """Возвращает блок контактов отдела соцвыплат для района из профиля."""
    d = (profile.get("Район") or "").lower()
    if not d:
        return ""
    for kw in DISTRICT_KEYS:
        if kw in d and kw in DISTRICT_OFFICES:
            return DISTRICT_OFFICES[kw]
    return ""

# ═══════════════════════════════════════════
#  РЕЕСТР МЕР — единый источник истины
#  Используется и для overview, и для проверки полноты.
#  Структура позволяет фильтровать по профилю программно.
# ═══════════════════════════════════════════

# Каждая мера: (название, краткое описание с точкой обращения)
MEASURES_UNIVERSAL = [
    ("Ежемесячная выплата (ЕДВ) 1 200 ₽", "на семью ежемесячно. Отдел соцвыплат района / МФЦ / Госуслуги"),
    ("Компенсация 30% на ЖКУ", "ежемесячно. Отдел соцвыплат района / МФЦ / Госуслуги"),
    ("Выплата на школьную форму 12 000 ₽", "на каждого школьника ежегодно. Отдел соцвыплат / МФЦ"),
    ("Компенсация платы за детсад", "20/50/70% за 1/2/3-го ребёнка. Детсад / Госуслуги"),
    ("Удостоверение многодетной семьи", "Отдел соцвыплат района / МФЦ / Госуслуги"),
    ("Бесплатный проезд школьников", "ЕСПБ / Карта жителя Кузбасса. Отдел соцвыплат района"),
    ("Первоочередной приём в детсад", "Управление образования / Госуслуги"),
    ("Льгота по транспортному налогу", "1 авто до 200 л.с. ФНС / Госуслуги"),
    ("Вычеты по налогам на имущество и землю", "+5/+7 кв.м на ребёнка, 6 соток. ФНС"),
    ("Стандартный вычет НДФЛ", "6 000 ₽/мес. за 3-го ребёнка. Через работодателя"),
    ("Областной маткапитал 130 000 ₽", "за 3-го ребёнка (после 90% федерального на жильё). Отдел соцвыплат"),
    ("Выплата 450 000 ₽ на ипотеку", "3-й ребёнок 2019–2030. Банк / Госуслуги / ДОМ.РФ"),
    ("Региональная выплата 550 000 ₽ на ипотеку", "3-й ребёнок 2026–2027. Банк / ДОМ.РФ"),
    ("Компенсация 50% за обучение в вузе/колледже", "Кузбасс. Через вуз/колледж"),
    ("Бесплатные пожарные извещатели", "Отдел соцвыплат района"),
    ("Бесплатная юрпомощь", "с 01.04.2025. Госюрбюро / КЦСОН"),
    ("Досрочная пенсия матери", "3 детей — 57 лет, 4 — 56, 5+ — 50. СФР"),
    ("Федеральный маткапитал", "СФР / Госуслуги (автоматически)"),
]
MEASURES_LOWINCOME = [
    ("Единое пособие на детей", "до 17 лет, 50/75/100% ПМ ребёнка (≈15 653 ₽). СФР / Госуслуги / МФЦ"),
    ("Семейная налоговая выплата", "возврат НДФЛ, с 01.06.2026, доход ≤ 1,5 ПМ. СФР / Госуслуги"),
    ("Ежеквартальная выплата", "только НЕПОЛНЫМ: 500/700/1000 ₽ за 3/4/5+ детей; полным — при 6+ детях. Отдел соцвыплат"),
    ("Бесплатное питание в школе", "Школа"),
    ("Бесплатные лекарства детям до 6 лет", "Поликлиника → аптека"),
    ("Социальный контракт", "приоритет многодетным. КЦСОН / Госуслуги"),
]

def build_registry(income_known_low=None):
    """
    Формирует реестр мер с фильтрацией по доходу.
    income_known_low:
      True  → доход ниже ПМ: показываем все меры
      False → доход выше ПМ: только меры без проверки дохода
      None  → неизвестен: все, с пометкой уточнить
    """
    lines = ["МЕРЫ БЕЗ ПРОВЕРКИ ДОХОДА (всем многодетным-гражданам РФ):"]
    for name, desc in MEASURES_UNIVERSAL:
        lines.append(f"• {name} — {desc}")
    if income_known_low is False:
        lines.append("\n(Семья указала доход выше ПМ — меры для малоимущих НЕ показываем и не упоминаем.)")
    else:
        header = "\nМЕРЫ ТОЛЬКО ДЛЯ МАЛОИМУЩИХ (доход ниже ПМ = 17 234 ₽/чел.):"
        if income_known_low is None:
            header += " — уточни доход пользователя, чтобы понять, что из этого доступно:"
        lines.append(header)
        for name, desc in MEASURES_LOWINCOME:
            lines.append(f"• {name} — {desc}")
    return "\n".join(lines)

# ═══════════════════════════════════════════
#  Классификация запроса (уточнённая)
# ═══════════════════════════════════════════

GREETING_RE = re.compile(r'^\s*(привет|здравствуй\w*|добр(ый|ое|ого) (день|утро|вечер|времени)|хай|хеллоу?|ку|доброго времени)[\s!.,)]*$', re.I)
# Благодарность: короткое сообщение, состоящее в основном из благодарственных слов
THANKS_WORDS = ("спасибо", "благодар", "спс", "пасибо", "понятно", "ясно", "окей", "хорошо")

def classify(message):
    m = message.lower().strip()
    if GREETING_RE.match(m) or (len(m) < 18 and any(w in m for w in ("привет", "здравств", "добрый день", "как дела"))):
        return "greeting"
    # благодарность: короткое сообщение (< 30 симв.) с благодарственным словом и без вопроса
    if len(m) < 30 and any(w in m for w in THANKS_WORDS) and "?" not in m and not any(
        x in m for x in ("льгот", "выплат", "пособи", "как ", "куда", "что ", "какие", "документ")):
        return "thanks"
    if re.match(r'^\s*(ок|ok|угу|ага)[\s!.,)]*$', m):
        return "thanks"
    # Обзор: спрашивают про всё сразу. Но НЕ если есть конкретная мера в запросе.
    specific_markers = ("жку", "коммунал", "пособи", "капитал", "ипотек", "налог", "проезд",
                        "питани", "садик", "сад", "форм", "удостоверен", "земл", "участок",
                        "лекарств", "пенси", "вуз", "колледж", "обучен", "контракт", "юрист",
                        "извещател", "документ", "как оформить", "как получить", "куда")
    overview_markers = ("какие льгот", "что положено", "что нам положено", "на что", "все льготы",
                       "что мне положено", "что я могу", "на какие", "перечень", "список льгот",
                       "что доступно", "какие выплаты", "что имеем", "на что имеем", "что полагается")
    has_overview = any(w in m for w in overview_markers)
    has_specific = any(w in m for w in specific_markers)
    if has_overview and not has_specific:
        return "overview"
    return "specific"

# ═══════════════════════════════════════════
#  Системный промпт (переработан под консультативный диалог)
# ═══════════════════════════════════════════

SYSTEM = """Ты — Анна, консультант центра социальной поддержки многодетных семей г. Кемерово. Ты опытный специалист по социальной работе: говоришь живым человеческим языком, по делу, с заботой, но без воды.

# ПРИВЕТСТВИЕ И ПОВТОРЫ (критично!)
- Здоровайся и представляйся («Здравствуйте! Я Анна…») ТОЛЬКО в самом первом сообщении диалога, когда история пуста. Если в истории уже есть твои реплики — НЕ здоровайся и НЕ представляйся заново, сразу отвечай по существу вопроса.
- НЕ пересказывай профиль семьи в каждом ответе («вижу, у вас трое детей 2, 15 и 19 лет…»). Учитывай профиль молча. Упоминай конкретную деталь профиля, только если она прямо влияет на ответ (например, право на меру зависит от возраста ребёнка или дохода).
- Не начинай ответы с шаблонных вводных вроде «Рада помочь вам разобраться». Сразу переходи к сути.

# ТВОЙ СТИЛЬ
- Пиши как живой человек, а не как справочник. Без канцелярита и шаблонов «(а)…(б)…».
- Отвечай ровно на то, что спросили. Не вываливай всё подряд — это раздражает и путает.
- Начинай с главного и самого ценного для семьи (деньги, крупные выплаты), мелочи — в конце или по запросу.
- Короткие уточняющие вопросы приветствуются — это суть консультации. Если для точного ответа не хватает данных (учится ли ребёнок очно, гражданство РФ, есть ли инвалидность) — спроси.
- Объясняй коротко «почему» — человеку важно понимать логику, а не только список.
- Длина ответа — по существу вопроса. На простой вопрос — 2-4 предложения. На «что положено» — структурированно, но без простыни.

# РАБОТА С ПРОФИЛЕМ СЕМЬИ
- Профиль семьи — НАДЁЖНЫЙ источник. Если он есть, верь ему больше, чем своим выводам или старым репликам в диалоге.
- Профиль говорит «3 детей» → семья многодетная, точка. Не пересчитывай, не спорь.
- Доход «выше ПМ» → меры для малоимущих просто НЕ упоминай (не пиши «вам не положено» — это обидно и лишне).
- Доход «ниже ПМ» → показывай и общие, и адресные меры.
- Доход «не знаю» → если мера зависит от дохода, мягко предупреди: «эта выплата — если доход ниже прожиточного минимума».
- Район указан → называй отдел соцвыплат именно этого района.

# ТОЧНОСТЬ (критично — это официальная информация)
- Суммы, адреса, документы, сроки — ТОЛЬКО из контекста базы знаний.
- НИКОГДА не выдумывай адреса и телефоны. Но и НЕ отправляй человека «уточнять» то, что есть в контексте.
- ДАВАЙ ПОЛНЫЙ ГОТОВЫЙ ОТВЕТ. По каждой мере раскрывай всё, что есть в контексте: что положено и сколько → какие документы нужны → куда подать (конкретный адрес, телефон, часы приёма) → как подать онлайн (Госуслуги/портал) → сроки рассмотрения. Не сворачивай ответ фразами «обратитесь и там подскажут», «уточните список документов на месте» — список и адрес уже в контексте, приведи их сам.
- Если в контексте есть блок «КОНТАКТЫ ОТДЕЛА СОЦВЫПЛАТ РАЙОНА ПОЛЬЗОВАТЕЛЯ» — указывай адрес/телефон/часы именно этого отдела. Не путай отдел соцвыплат с МФЦ — это разные учреждения с разными адресами.
- Не пиши «обратитесь в УСЗН» — это орган управления, туда не ходят. Конкретно: отдел соцвыплат района, клиентская служба СФР, МФЦ, школа, детсад, ФНС, вуз.
- Полезные детали, о которых люди забывают: выплаты идут на карту МИР; пособия не назначают «задним числом»; у справок есть срок действия. Упоминай к месту.
- С 01.03.2026 региональные меры Кузбасса — только для граждан РФ.

# ГРАНИЦЫ
- Вопрос не про льготы/соцподдержку (погода, политика, рецепты) — мягко верни в тему: «Я консультирую по льготам для многодетных семей Кемерово, тут помогу с радостью». Не выдумывай ответы вне темы.
- Перенаправляй на горячую линию (СФР 8-800-100-00-01, dsznko.ru) ТОЛЬКО если конкретного факта реально нет в контексте — и тогда сначала дай всё, что есть, а телефон укажи лишь для недостающей детали. Не используй перенаправление вместо ответа.

Отвечай на русском. Ключевое выделяй **жирным**. Тон — тёплый, профессиональный, без формализма."""


class ChatRequest(BaseModel):
    message: str
    history: list = []
    profile: dict = {}


def parse_income(profile):
    """Возвращает True (ниже ПМ), False (выше ПМ), None (неизвестно)."""
    d = (profile.get("Доход") or "").lower()
    if "ниже" in d:
        return True
    if "выше" in d:
        return False
    return None


def build_profile_block(profile):
    p = []
    if profile.get("Детей"): p.append(f"детей: {profile['Детей']}")
    if profile.get("Возраст"): p.append(f"возраст детей: {profile['Возраст']}")
    if profile.get("Семья"): p.append(f"состав: {profile['Семья'].lower()}")
    inc = parse_income(profile)
    if inc is True: p.append("доход НИЖЕ прожиточного минимума")
    elif inc is False: p.append("доход ВЫШЕ прожиточного минимума")
    if profile.get("Ипотека") == "Да": p.append("есть ипотека")
    if profile.get("Район"): p.append(f"район Кемерово: {profile['Район']}")
    if not p:
        return "Профиль семьи не заполнен — при необходимости уточни недостающее у пользователя."
    return "ПРОФИЛЬ СЕМЬИ (надёжные данные, верь им): " + ", ".join(p) + "."


# Простое логирование для метрик (проблема А3) — пишем в файл
LOG_PATH = "/tmp/consultant_log.jsonl"
def log_interaction(message, kind, n_chunks, profile):
    try:
        rec = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "msg": message[:200],
            "kind": kind,
            "chunks": n_chunks,
            "has_profile": bool(profile.get("Детей")),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def call_gemini(messages: list, system: str) -> str:
    """Вызов Google Gemini (google-genai SDK). Основной провайдер."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or not GEMINI_AVAILABLE:
        raise RuntimeError("GEMINI_API_KEY не задан или google-genai не установлен")

    client = google_genai.Client(api_key=api_key)

    # Собираем историю в формат Gemini (role: user/model)
    contents = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            continue  # system идёт отдельно
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config={
            "system_instruction": system,
            "temperature": 0.2,
            "max_output_tokens": 2048,
            # gemini-2.5-flash — «думающая» модель: по умолчанию reasoning-токены
            # съедают бюджет max_output_tokens, из-за чего видимый ответ
            # обрывается на полуслове. Для фактологического консультанта
            # пошаговое мышление не нужно — отключаем (budget=0).
            "thinking_config": {"thinking_budget": 0},
        }
    )
    # При обрыве по лимиту/блокировке response.text может быть пустым —
    # бросаем исключение, чтобы сработал резервный Groq, а не пустой ответ.
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini вернул пустой ответ (возможно, обрыв по токенам)")
    return text


def call_groq(messages: list) -> str:
    """Вызов Groq Llama. Резервный провайдер."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or not GROQ_AVAILABLE:
        raise RuntimeError("GROQ_API_KEY не задан или groq не установлен")

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=2048,
        top_p=0.9,
    )
    return resp.choices[0].message.content


def call_llm(messages: list, system: str) -> str:
    """
    Двухуровневый вызов LLM:
    1. Пробуем Groq (ОСНОВНОЙ — быстрый, щедрые лимиты, без обрыва ответов)
    2. При ошибке/rate-limit — автоматически переключаемся на Gemini (резерв)
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    # Нет ни одного ключа — честное сообщение
    if not groq_key and not gemini_key:
        return ("⚠️ Не задан ни один API-ключ. Добавьте в Render:\n"
                "• GROQ_API_KEY (основной: console.groq.com)\n"
                "• GEMINI_API_KEY (резерв: aistudio.google.com)")

    # Пробуем Groq если ключ есть
    if groq_key and GROQ_AVAILABLE:
        try:
            return call_groq(messages)
        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ("rate_limit", "429", "rate", "too many")):
                print(f"[Groq] rate limit, переключаюсь на Gemini: {err[:80]}")
                # fallthrough к Gemini ниже
            elif any(x in err.lower() for x in ("invalid_api_key", "authentication", "auth")):
                print(f"[Groq] неверный ключ, пробую Gemini: {err[:80]}")
            else:
                print(f"[Groq] ошибка: {err[:150]}")

    # Gemini как резерв
    if gemini_key and GEMINI_AVAILABLE:
        try:
            return call_gemini(messages, system)
        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ("429", "quota", "rate", "resource_exhausted")):
                return "⏳ Оба сервиса временно перегружены. Подождите 30 секунд и повторите."
            if any(x in err.lower() for x in ("api key", "permission", "authentication")):
                return "⚠️ Неверный GEMINI_API_KEY. Проверьте ключ в Render Environment."
            return f"⚠️ Ошибка резервного провайдера: {err[:150]}"

    return "⚠️ Не удалось подключиться ни к одному провайдеру. Проверьте ключи."


@app.post("/api/chat")
async def chat(req: ChatRequest):
    kind = classify(req.message)
    income_low = parse_income(req.profile)
    profile_block = build_profile_block(req.profile)

    # ── Формируем контекст по типу запроса ──
    if kind == "greeting":
        context = "(Пользователь поздоровался. Поздоровайся в ответ коротко и тепло, представься как консультант по льготам для многодетных семей, спроси чем помочь. НЕ перечисляй льготы.)"
        n_chunks = 0
    elif kind == "thanks":
        context = "(Пользователь благодарит или подтверждает. Ответь коротко и доброжелательно, предложи помочь с чем-то ещё. НЕ повторяй список льгот.)"
        n_chunks = 0
    elif kind == "overview":
        registry = build_registry(income_low)
        rag = retrieve(req.message, top_k=2)
        context = (registry +
                   ("\n\nДЕТАЛИ ИЗ БАЗЫ:\n" + "\n---\n".join(rag) if rag else "") +
                   "\n\n(Это полный реестр. Не вываливай всё списком — выдели 4-6 самых важных и ценных мер для этой семьи, сгруппируй понятно, предложи углубиться в любую. Начни с денежных выплат.)")
        n_chunks = len(rag)
    else:  # specific
        if not is_on_topic(req.message):
            context = "(Вопрос НЕ относится к льготам и соцподдержке. Мягко верни пользователя в тему — ты консультант по льготам для многодетных семей Кемерово. Не отвечай на посторонний вопрос по существу.)"
            n_chunks = 0
        else:
            rag = retrieve(req.message, top_k=6)
            if not rag:
                context = "(В базе нет точных данных по этому вопросу. Честно скажи это, направь на горячую линию СФР 8-800-100-00-01 или dsznko.ru. Не выдумывай.)"
            else:
                context = "\n\n---\n\n".join(rag)
            n_chunks = len(rag)

    # ── Детерминированно подкладываем контакты отдела района пользователя ──
    # (только для содержательных запросов, не для приветствия/благодарности/оффтопа)
    if kind in ("overview", "specific") and n_chunks > 0:
        cb = district_contact_block(req.profile)
        if cb:
            context = ("СПРАВОЧНО — отдел соцвыплат района пользователя. Используй "
                       "эти точные адрес/телефон/часы, ТОЛЬКО если мера оформляется "
                       "в отделе соцвыплат (региональные: ЖКУ, ЕДВ, удостоверение, "
                       "школьная форма, проездной, ежеквартальная, обл. маткапитал). "
                       "Для федеральных мер веди в СФР, для налогов — в ФНС, для "
                       "питания — в школу. Не путай отдел соцвыплат с МФЦ.\n" + cb +
                       "\n\n---\n\n" + context)

    log_interaction(req.message, kind, n_chunks, req.profile)

    system_full = SYSTEM + "\n\n# ТЕКУЩИЙ ПОЛЬЗОВАТЕЛЬ\n" + profile_block
    messages = [{"role": "system", "content": system_full}]
    for pair in req.history[-3:]:
        if pair.get("user"):
            messages.append({"role": "user", "content": pair["user"]})
        if pair.get("bot"):
            messages.append({"role": "assistant", "content": pair["bot"]})
    messages.append({"role": "user",
                     "content": f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context}\n\nВОПРОС: {req.message}"})

    answer = call_llm(messages, system_full)
    return {"answer": answer}


@app.get("/health")
async def health():
    return {"status": "ok", "chunks": len(CHUNKS)}

@app.get("/metrics")
async def metrics():
    """Простые метрики для оценки (проблема А3)."""
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        kinds = {}
        for r in lines:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        no_context = sum(1 for r in lines if r["kind"] == "specific" and r["chunks"] == 0)
        return {"total": len(lines), "by_kind": kinds, "specific_no_context": no_context}
    except FileNotFoundError:
        return {"total": 0}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_frontend(full_path: str):
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())
