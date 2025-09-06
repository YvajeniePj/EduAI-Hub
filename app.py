import json, re, os
import streamlit as st
import io, pandas as pd
from dotenv import load_dotenv
from statistics import mean

load_dotenv()

from openai import OpenAI

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = None
if DEEPSEEK_API_KEY:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")  # OpenAI-compatible
DEEPSEEK_MODEL = "deepseek-chat"  # or "deepseek-reasoner"


# --- persistence + badges
POINTS_PATH = "points.json"
SUBM_PATH   = "submissions.json"
REVIEWS_PATH = "reviews.json"

def current_subject() -> str:
    return st.session_state.get("subject", "General")

def assignment_selector(widget_key: str):
    subj = current_subject()
    sub_rubrics = [r for r in RUBRICS if r.get("subject", "General") == subj]
    if not sub_rubrics:
        st.warning("Для этого предмета пока нет заданий.")
        st.stop()
    id_to_r = {r["assignment_id"]: r for r in sub_rubrics}
    ids = list(id_to_r.keys())

    cur = st.session_state.get("assign_id", ids[0])
    if cur not in ids:
        cur = ids[0]

    rid = st.selectbox(
        "Задание:",
        options=ids,
        index=ids.index(cur),
        format_func=lambda rid: id_to_r[rid]["title"],
        key=widget_key,
    )
    st.session_state["assign_id"] = rid
    return id_to_r[rid]

def submission_preview(s: dict, max_len: int = 60) -> str:
    """Короткий текст для списка выбора в кросс-проверке.
    Поддерживает старые ('answer') и новые ('answers'=[...]) форматы."""
    if "answer" in s:  # старый формат
        txt = s.get("answer") or ""
    elif "answers" in s and s["answers"]:
        # возьмём первый ответ, либо склеим первые строки
        parts = [a.get("answer", "") for a in s["answers"] if a.get("answer")]
        txt = " | ".join(parts[:2]) if parts else ""
    else:
        txt = ""
    txt = txt.strip().replace("\n", " ")
    return (txt[:max_len] + "…") if len(txt) > max_len else txt


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def badge_for(points: int) -> str:
    if points >= 50: return "🏅 Power Learner"
    if points >= 30: return "🥇 Consistent"
    if points >= 15: return "🥈 Getting There"
    if points >= 5:  return "🥉 Starter"
    return "🌱 Rookie"

def migrate_stores_to_subject_scope():
    # был dict user->points, теперь dict subject->(dict user->points)
    if isinstance(st.session_state.points, dict) and st.session_state.points and \
       all(isinstance(v, int) for v in st.session_state.points.values()):
        st.session_state.points = {"General": st.session_state.points}
        save_json(POINTS_PATH, st.session_state.points)

    # dict subject->list
    if isinstance(st.session_state.submissions, list):
        st.session_state.submissions = {"General": st.session_state.submissions}
        save_json(SUBM_PATH, st.session_state.submissions)

    # dict subject->list
    if isinstance(st.session_state.reviews, list):
        st.session_state.reviews = {"General": st.session_state.reviews}
        save_json(REVIEWS_PATH, st.session_state.reviews)

def get_points_store() -> dict:
    subj = current_subject()
    if subj not in st.session_state.points:
        st.session_state.points[subj] = {}
    return st.session_state.points[subj]

def save_points_store():
    save_json(POINTS_PATH, st.session_state.points)

def get_submissions_store() -> list:
    subj = current_subject()
    if subj not in st.session_state.submissions:
        st.session_state.submissions[subj] = []
    return st.session_state.submissions[subj]

def save_submissions_store():
    save_json(SUBM_PATH, st.session_state.submissions)

def get_reviews_store() -> list:
    subj = current_subject()
    if subj not in st.session_state.reviews:
        st.session_state.reviews[subj] = []
    return st.session_state.reviews[subj]

def save_reviews_store():
    save_json(REVIEWS_PATH, st.session_state.reviews)



st.set_page_config(page_title="EduAI Hub (Mini)", page_icon="🎓", layout="wide")
st.markdown("""
<style>
/* Layout spacing */
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

/* Cards */
.card {
  border: 1px solid #eaeaea;
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  background: #fff;
  margin-bottom: 12px;
}

/* Buttons */
div.stButton > button {
  border-radius: 10px;
  padding: 0.55rem 1rem;
  border: 1px solid #e5e7eb;
}

/* Badges (chips) */
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: #eef2ff;
  color: #3730a3;
  font-size: 12px;
  border: 1px solid #e0e7ff;
}

/* Muted small text */
.muted { color: #6b7280; font-size: 12px; }

/* Tables (slightly denser) */
.dataframe tbody tr th, .dataframe tbody tr td { padding: 6px 10px; }
</style>
""", unsafe_allow_html=True)

st.title("EduAI Hub — платформа для эффективного обучения")
st.divider()


# --- tiny "db" with persistence
if "points" not in st.session_state:
    st.session_state.points = load_json(POINTS_PATH, {})
if "submissions" not in st.session_state:
    st.session_state.submissions = load_json(SUBM_PATH, [])
if "reviews" not in st.session_state:
    st.session_state.reviews = load_json(REVIEWS_PATH, [])

migrate_stores_to_subject_scope()

# --- load rubric
with open("rubric.json", "r", encoding="utf-8") as f:
    RUBRICS = json.load(f)   # list of rubrics (>=1)
id_to_rubric = {r["assignment_id"]: r for r in RUBRICS}

def keyword_score_for(text: str, kw_list: list[dict], max_points: int):
    text_lc = (text or "").lower()
    score = 0
    details = []
    for kw in kw_list:
        found = bool(re.search(rf"\b{re.escape(kw['word'].lower())}\b", text_lc))
        if found:
            score += kw["points"]
            details.append(f"+{kw['points']}: найдено '{kw['word']}'")
        else:
            details.append(f"+0: нет '{kw['word']}'")
    score = min(score, max_points)
    return score, details


def llm_grade(answer_text: str, rubric: dict) -> dict | None:
    """
    Ask DeepSeek to grade 0–100 and give short feedback bullets.
    Returns {"score": int, "feedback": [str, ...]} or None on failure.
    """
    if client is None:
        st.warning("AI-оценка: не найден ключ (DEEPSEEK_API_KEY).")
        return None
    try:
        system_msg = (
            "You are a strict but fair TA. "
            "Grade the student's short answer on a 0–100 scale based ONLY on the rubric. "
            "Return strict JSON: {\"score\": <int>, \"feedback\": [\"...\", \"...\"]}"
        )
        user_msg = (
            "Rubric title: " + rubric.get("title","") + "\n"
            "Max points: 100\n"
            "Keywords (hints): " + ", ".join([k['word'] for k in rubric.get('keywords', [])]) + "\n"
            "Student answer:\n" + answer_text
        )

        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
        )
        content = resp.choices[0].message.content.strip()

        # Extract JSON
        import json, re
        m = re.search(r"\{.*\}", content, re.S)
        json_str = m.group(0) if m else content
        data = json.loads(json_str)

        score = int(max(0, min(100, data.get("score", 0))))
        feedback = data.get("feedback", [])
        if not isinstance(feedback, list): feedback = [str(feedback)]
        return {"score": score, "feedback": feedback[:5]}

    except Exception as e:
        # Surface the real reason in the UI so you know what to fix
        st.error(f"AI ошибка: {getattr(e, 'message', str(e))[:200]}")
        return None

def award_points(user: str, pts: int):
    pts_store = get_points_store()
    pts_store[user] = pts_store.get(user, 0) + pts

def peer_avg_for_submission(subm_idx: int):
    reviews = get_reviews_store()
    scores = [r["avg_score"] for r in reviews if r["submission_idx"] == subm_idx]
    return (round(mean(scores), 2), len(scores)) if scores else (None, 0)

# --- sidebar "login"
st.sidebar.markdown("### ⚙️ Настройки")
st.sidebar.header("Профиль")
user = st.sidebar.text_input("Ваше имя (для лидерборда)", value="Student")

# --- subject picker
SUBJECTS = sorted({ r.get("subject", "General") for r in RUBRICS })
subject = st.sidebar.selectbox("Предмет", SUBJECTS, key="subject")


use_ai = st.sidebar.checkbox("Включить AI-оценку (опционально)", value=True)
st.session_state["use_ai"] = bool(use_ai)  # <-- ВАЖНО: кладём флаг в session_state

# Connection status
if client is None:
    st.sidebar.error("AI отключен: нет DEEPSEEK_API_KEY в .env")
else:
    try:
        _ = client.models.list()  # ping совместимого API
        st.sidebar.success("AI подключен (DeepSeek)")
    except Exception as e:
        st.sidebar.error(f"AI недоступен: {getattr(e, 'message', str(e))[:120]}")
st.sidebar.info("Pегистрация не требуется.")


st.sidebar.divider()
st.sidebar.markdown("**О проекте**\n\nМини-платформа для проверки ДЗ и кросс-оценки. Экспорт в CSV, бейджи за активность.")

tab_submit, tab_leaderboard, tab_peer, tab_chat = st.tabs(
    ["📝 Мои задания", "🏆 Лидерборд", "🤝 Кросс-проверка", "💬 Чат-ассистент (демо)"]
)

with tab_peer:

    st.subheader("Кросс-проверка (анонимно)")
    st.subheader("Выбор задания")
    RUBRIC = assignment_selector("assign_id_peer")
    st.subheader("Ответы на вопросы задания")

    # Pick which assignment to review (reuse current selection)
    st.caption("Выберите задание для проверки (совпадает с селектором выше).")
    # We already have assign_id and RUBRIC bound from the selector section.

    # Build candidate submissions
    subs = get_submissions_store()
    all_subms = [
        (idx, s) for idx, s in enumerate(subs)
        if s.get("assignment") == RUBRIC["assignment_id"] and s.get("user") != user
    ]

    if not all_subms:
        st.info("Пока нет чужих ответов по этому заданию.")
    else:
        # Show a simple picker
        option_indices = [idx for idx, _ in all_subms]
        labels = [f"#{idx} — {s.get('user','?')}: {submission_preview(s)}" for idx, s in all_subms]
        sel = st.selectbox("Выберите ответ для проверки:", options=range(len(option_indices)),
                        format_func=lambda i: labels[i])
        pick_idx = option_indices[sel]
        pick = subs[pick_idx]


        st.write("Ответ студента")
        if "answer" in pick:
            # старый формат
            st.write(pick["answer"])
        elif "answers" in pick and pick["answers"]:
            # новый формат: покажем каждый вопрос отдельным блоком
            for ans in pick["answers"]:
                qid = ans.get("question_id", "")
                atxt = (ans.get("answer") or "").strip()
                st.markdown(f"**{qid}**")
                st.write(atxt if atxt else "—")
                st.markdown("<hr style='border:none;border-top:1px solid #eee;margin:8px 0;' />", unsafe_allow_html=True)
        else:
            st.write("Ответ отсутствует.")


        st.markdown("### Оцените по критериям (1–5)")
        c1, c2 = st.columns(2)
        with c1:
            sc_relevance  = st.slider("Соответствие заданию", 1, 5, 3)
            sc_structure  = st.slider("Структура и логика", 1, 5, 3)
        with c2:
            sc_argument   = st.slider("Аргументация / примеры", 1, 5, 3)
            sc_clarity    = st.slider("Ясность изложения", 1, 5, 3)
        comment = st.text_area("Короткий комментарий (опционально)", placeholder="Что можно улучшить?")

        avg_score = round((sc_relevance + sc_structure + sc_argument + sc_clarity) / 4, 2)
        st.info(f"Средняя оценка по критериям: **{avg_score} / 5**")

        if st.button("Отправить отзыв"):
            review = {
                "submission_idx": pick_idx,
                "assignment": RUBRIC["assignment_id"],
                "reviewer": user,          
                "scores": {
                    "relevance": sc_relevance,
                    "structure": sc_structure,
                    "argument": sc_argument,
                    "clarity": sc_clarity
                },
                "avg_score": avg_score,
                "comment": comment.strip()
            }
            reviews = get_reviews_store()
            reviews.append(review)
            save_reviews_store()

            award_points(user, 1)
            save_points_store()
            st.success("Отзыв сохранен! Вам начислено +1 очко за кросс-проверку.")

    st.markdown("---")
    st.subheader("Мои полученные отзывы")
    subs = get_submissions_store()
    my_subms = [
        (i, s) for i, s in enumerate(subs)
        if s.get("assignment") == RUBRIC["assignment_id"] and s.get("user") == user
    ]
    reviews = get_reviews_store()

    if not my_subms:
        st.write("Вы ещё не сдавали это задание.")
    else:
        for idx, s in my_subms:
            avg, cnt = peer_avg_for_submission(idx)
            st.markdown(
                f"**Сдача #{idx}** — peer-оценок: {cnt}"
                + (f", среднее: **{avg}/5**" if avg is not None else "")
            )
            # последние комментарии по этой сдаче
            comments = [
                r.get("comment") for r in reviews
                if isinstance(r, dict)
                and r.get("submission_idx") == idx
                and r.get("comment")
            ]
            if comments:
                with st.expander("Комментарии"):
                    for c in comments[-5:]:
                        st.write("•", c)


with tab_submit:

    st.subheader("Выбор задания")
    RUBRIC = assignment_selector("assign_id_submit")

    # RUBRIC is bound from the assignment selector (as before)
    questions = RUBRIC.get("questions", [])
    if not questions:
        st.info("Для этого задания ещё не настроены вопросы.")
    else:
        # Render a text area for each question
        answers = {}
        for q in questions:
            st.markdown(f"### {q['title']}  \n*Макс. баллов: {q['max_points']}*")
            answers[q["question_id"]] = st.text_area(
                f"Ваш ответ ({q['question_id']})",
                key=f"ans_{RUBRIC['assignment_id']}_{q['question_id']}",
                height=140
            )
            st.markdown("---")

        if st.button("Проверить все"):
            total_score = 0
            total_max = sum(q["max_points"] for q in questions)
            per_q_results = []

            # Run baseline keyword scoring per question
            for q in questions:
                ans_text = answers.get(q["question_id"], "")
                q_score_kw, q_details = keyword_score_for(
                    ans_text, q["keywords"], q["max_points"]
                )
                per_q_results.append({
                    "question_id": q["question_id"],
                    "title": q["title"],
                    "answer": ans_text,
                    "kw_score": q_score_kw,
                    "details": q_details
                })
                total_score += q_score_kw

            # Optional AI refinement (average each question’s score with AI)
            use_ai = st.session_state.get("use_ai")  # if you stored it; else read from sidebar
            if 'use_ai' not in st.session_state:
                # if you defined the checkbox in the sidebar, you likely have a local variable 'use_ai'
                # fallback: try to use that local, else leave None
                try:
                    use_ai_local = use_ai
                except:
                    use_ai_local = False
            else:
                use_ai_local = st.session_state['use_ai']

            ai_total_delta = 0
            if use_ai_local:
                for item in per_q_results:
                    if client is None:
                        continue
                    # Build a temporary one-question rubric for llm_grade
                    one_q_rubric = {
                        "title": f"{RUBRIC['title']} — {item['title']}",
                        "keywords": [{"word": k["word"]} for k in next(q for q in questions if q["question_id"]==item["question_id"])["keywords"]]
                    }
                    with st.spinner(f"AI оценивает: {item['question_id']}"):
                        llm_info = llm_grade(item["answer"], one_q_rubric)
                    if llm_info:
                        # Combine keyword score (scaled to 0–100 by question max) with AI score (0–100)
                        # then rescale back to question max
                        kw_pct = (item["kw_score"] / next(q for q in questions if q["question_id"]==item["question_id"])["max_points"]) * 100
                        combined_pct = (kw_pct + llm_info["score"]) / 2
                        combined_score = round(combined_pct / 100 * next(q for q in questions if q["question_id"]==item["question_id"])["max_points"])
                        ai_total_delta += (combined_score - item["kw_score"])
                        item["ai_score"] = int(round(llm_info["score"]))
                        item["final_score"] = combined_score
                        item["ai_feedback"] = llm_info.get("feedback", [])
                    else:
                        item["final_score"] = item["kw_score"]
                total_score += ai_total_delta  # adjust total if AI used
            else:
                for item in per_q_results:
                    item["final_score"] = item["kw_score"]

            # Show per-question breakdown
            st.success(f"Итог: {total_score}/{total_max}")
            for item in per_q_results:
                with st.expander(f"{item['question_id']}: {item['title']} — {item['final_score']} баллов"):
                    st.markdown("**По ключевым словам:**")
                    for d in item["details"]:
                        st.write("•", d)
                    if item.get("ai_feedback"):
                        st.markdown("**AI-фидбек:**")
                        for fb in item["ai_feedback"]:
                            st.write("•", fb)

            # Award points (same rule: 1 балл за каждые ~10% итога, минимум 1)
            pct = total_score / max(1, total_max)
            gained = max(1, int(round(pct * 10)))
            award_points(user, gained)
            save_points_store()

            # Save a multi-question submission
            subs = get_submissions_store()
            subs.append({
                "user": user,
                "assignment": RUBRIC["assignment_id"],
                "answers": [
                    {
                        "question_id": item["question_id"],
                        "answer": item["answer"],
                        "score": item["final_score"]
                    } for item in per_q_results
                ],
                "total_score": total_score,
                "total_max": total_max,
                "points_awarded": gained
            })
            save_submissions_store()

            st.info(f"Начислено {gained} очк.")


with tab_leaderboard:
    st.subheader(f"Лидерборд — {current_subject()}")
    pts = get_points_store()
    if not pts:
        st.write("Пока пусто.")
    else:
        board = sorted(pts.items(), key=lambda x: x[1], reverse=True)
        for i, (u, p) in enumerate(board, start=1):
            st.write(f"{i}. **{u}** — {p} очк. {badge_for(p)}")

     # описание уровней
    st.markdown("### 🏅 Значки и уровни")
    st.markdown("""
    - 🌱 **Rookie** — 0–4 очка. Начало пути!
    - 🥉 **Starter** — 5–14 очков. Первые шаги.
    - 🥈 **Getting There** — 15–29 очков. Хороший прогресс.
    - 🥇 **Consistent** — 30–49 очков. Стабильные результаты.
    - 🏅 **Power Learner** — 50+ очков. Отличная вовлечённость!
    """)

    subs = get_submissions_store()
    with st.expander("Последние сдачи"):
        if not subs:
            st.write("Нет сдач.")
        else:
            for s in reversed(subs[-10:]):
                summary = f"{s['total_score']}/{s['total_max']}" if "total_score" in s else str(s.get("score","?"))
                preview = submission_preview(s, max_len=40)
                st.markdown(f"- **{s['user']}** → {summary}  <span class='muted'>(+{s['points_awarded']} очк.)</span>  — {preview}", unsafe_allow_html=True)

    # Экспорт по предмету
    pts = get_points_store()
    subs = get_submissions_store()
    revs = get_reviews_store()

    if pts:
        df_points = pd.DataFrame([{"user": u, "points": p} for u, p in pts.items()]).sort_values(
            "points", ascending=False
        )
    else:
        df_points = pd.DataFrame(columns=["user", "points"])

    df_subm = pd.DataFrame(subs) if subs else pd.DataFrame(columns=[
        "user", "assignment", "answers", "total_score", "total_max", "points_awarded"
    ])

    df_rev = pd.DataFrame(revs) if revs else pd.DataFrame(columns=[
        "submission_idx", "assignment", "reviewer", "scores", "avg_score", "comment"
    ])

    st.write("—")
    st.markdown("**Выгрузка данных**")
    buf1, buf2, buf3 = io.BytesIO(), io.BytesIO(), io.BytesIO()
    df_points.to_csv(buf1, index=False, encoding="utf-8-sig")
    df_subm.to_csv(buf2, index=False, encoding="utf-8-sig")
    df_rev.to_csv(buf3, index=False, encoding="utf-8-sig")
    st.download_button("⬇️ Лидерборд (CSV)", data=buf1.getvalue(),
                    file_name=f"leaderboard_{current_subject()}.csv", mime="text/csv")
    st.download_button("⬇️ Сдачи (CSV)", data=buf2.getvalue(),
                    file_name=f"submissions_{current_subject()}.csv", mime="text/csv")
    st.download_button("⬇️ Отзывы (CSV)", data=buf3.getvalue(),
                    file_name=f"reviews_{current_subject()}.csv", mime="text/csv")


with tab_chat:
    st.subheader("FAQ-ассистент (демо)")
    st.caption("Простой демон без LLM: отвечает на типовые вопросы курса.")
    q = st.text_input("Вопрос:")
    faq = {
        "дедлайн": "Дедлайн сегодня в 23:59 (демо).",
        "формат": "Ответ 3–10 предложений, PDF не нужен (демо).",
        "правила": "Плагиат запрещен. Разрешена кросс-проверка (демо)."
    }
    if st.button("Спросить"):
        a = None
        for k, v in faq.items():
            if k in q.lower():
                a = v
                break
        if a is None:
            a = "Пока знаю только ответы про: " + ", ".join(faq.keys())
        st.write("**Ответ:**", a)

st.caption("⚙️ Минимальный прототип: ключевые слова → оценка, очки → лидерборд, простой FAQ.")

