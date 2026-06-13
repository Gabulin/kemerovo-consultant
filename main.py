"""
RAG-консультант по льготам для многодетных семей г. Кемерово
Backend: FastAPI  |  Python 3.11  |  Groq API (Llama 3.3 70B)
"""

import os, re, math, json
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# ═══════════════════════════════════════════
#  RAG: загрузка и индексация базы знаний
# ═══════════════════════════════════════════

KB_FILES = [
    "kb/01_federal.md",
    "kb/02_regional.md",
    "kb/03_special.md",
    "kb/04_organizations.md",
    "kb/05_documents.md",
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

def retrieve(query, top_k=7):
    qt = tok(query + " льготы многодетные Кемерово выплаты документы")
    qtf = {}
    for t in qt:
        qtf[t] = qtf.get(t, 0) + 1
    total = sum(qtf.values()) or 1
    qvec = {t: (qtf[t] / total) * IDF.get(t, 1) for t in qtf}
    scores = sorted(enumerate(VECS), key=lambda x: cos(qvec, x[1]), reverse=True)
    results, seen = [], set()
    for idx, _ in scores:
        if cos(qvec, VECS[idx]) < 0.005:
            break
        key = CHUNKS[idx][:100]
        if key not in seen:
            seen.add(key)
            results.append(CHUNKS[idx])
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
print(f"[RAG] Индекс готов: {len(IDF)} токенов")

# ═══════════════════════════════════════════
#  Системный промпт
# ═══════════════════════════════════════════

SYSTEM = """Ты — консультант по льготам для многодетных семей г. Кемерово (Кузбасс). Отвечай строго по контексту.

ПРАВИЛА:
1. Используй ТОЛЬКО данные из контекста — не придумывай суммы, адреса, законы.
2. Разделяй: (а) льготы всем многодетным; (б) только малоимущим (доход ниже ПМ).
3. По каждой мере: что даёт → документы → куда идти в Кемерово (адрес, телефон, часы) → онлайн → срок → закон.
4. Никогда не пиши «обратитесь в УСЗН» — называй конкретное учреждение (отдел соцвыплат района, СФР, МФЦ).
5. При разговорных вопросах («что мне дадут», «куда за деньгами») — понимай намерение, отвечай по сути.
6. Учитывай профиль семьи если указан.
7. С 01.03.2026 региональные меры — только гражданам РФ.
8. Если данных нет — скажи честно: dsznko.ru, СФР 8-800-100-00-01, +7 (3842) 58-33-46.
9. Отвечай на русском. Используй **жирный** для выделения ключевого."""

# ═══════════════════════════════════════════
#  API эндпоинты
# ═══════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    history: list = []
    profile: dict = {}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {"answer": "⚠️ GROQ_API_KEY не задан. Добавьте переменную окружения на Render."}

    chunks = retrieve(req.message)
    if not chunks:
        return {"answer": "По вашему вопросу данных не найдено.\n\nОбратитесь: СФР **8-800-100-00-01** или **dsznko.ru**"}

    ctx = "\n\n---\n\n".join(chunks)
    p = {k: v for k, v in req.profile.items() if v and v not in ("Не знаю", "Нет")}
    profile_str = ("\n\nПРОФИЛЬ СЕМЬИ: " + " | ".join(f"{k}: {v}" for k, v in p.items())) if p else ""
    user_content = f"КОНТЕКСТ:\n{ctx}{profile_str}\n\nВОПРОС: {req.message}"

    messages = [{"role": "system", "content": SYSTEM}]
    for pair in req.history[-3:]:
        if pair.get("user"):
            messages.append({"role": "user", "content": pair["user"]})
        if pair.get("bot"):
            messages.append({"role": "assistant", "content": pair["bot"]})
    messages.append({"role": "user", "content": user_content})

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.15,
            max_tokens=1800,
        )
        return {"answer": resp.choices[0].message.content}
    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower():
            return {"answer": "⏳ Лимит запросов. Подождите 1 минуту."}
        if "invalid_api_key" in err.lower() or "authentication" in err.lower():
            return {"answer": "⚠️ Неверный GROQ_API_KEY. Проверьте переменную окружения."}
        return {"answer": f"⚠️ Ошибка: {err[:150]}"}

@app.get("/health")
async def health():
    return {"status": "ok", "chunks": len(CHUNKS)}

# Отдаём index.html для всех остальных маршрутов
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_frontend(full_path: str):
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())
