# app.py
import streamlit as st
import os, json, time, re, hashlib, tempfile
import nest_asyncio
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Fix asyncio for Streamlit ----------------
nest_asyncio.apply()

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs_local")
HIST_DIR = os.path.join(BASE_DIR, "history")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

# ---------------- Persistent History ----------------
def hist_path(username: str, password: str): 
    safe_username = "".join(c for c in username if c.isalnum() or c in ("-", "_")).strip() or "anonymous"
    hashed_pw = hashlib.sha256(password.encode("utf-8")).hexdigest()
    filename = f"{safe_username}_{hashed_pw}.json"
    return os.path.join(HIST_DIR, filename)

def load_history(username: str, password: str):
    try:
        with open(hist_path(username, password), "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_history(username: str, password: str, records):
    try:
        with open(hist_path(username, password), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save history: {e}")

# ---------------- Cleanup ----------------
def clean_text(text: str):
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r"(This post was edited|on Reddit|Thanks for the reply|Posted by)", line, re.IGNORECASE):
            continue
        if re.search(r"\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?", line):
            continue
        if len(line.split()) <= 2:
            continue
        clean_lines.append(line)
    return " ".join(clean_lines)

# ---------------- Model Loading ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    st.info("Loading Granite 3.2 2B Instruct from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm-granite/granite-3.2-2b-instruct",
        token="#######"  # Replace with your Hugging Face token
    )
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.2-2b-instruct",
        token="#######3"
    )
    st.success("Granite 3.2 2B Instruct loaded ✅")
    return tokenizer, model, "Granite-3.2-2B-Instruct"

tokenizer, model, model_type = load_model()

# ---------------- Rewrite Function ----------------
def rewrite_text(text: str, tone: str) -> str:
    prompt = f"{text}\n\nRewrite this in a {tone} tone. Only output the rewritten text."

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        num_return_sequences=1,
        repetition_penalty=2.5,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.8,
        temperature=0.7
    )

    rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-processing: remove greetings or repeated letters
    rewritten_text = re.sub(r"(Hello,? how can I help.*?(\.|!))", "", rewritten_text, flags=re.IGNORECASE)
    rewritten_text = re.sub(r"(.)\1{2,}", r"\1", rewritten_text)
    rewritten_text = " ".join(rewritten_text.split())

    return rewritten_text

# ---------------- Edge TTS Audio ----------------
def generate_audio(text, voice_type, filename):
    voice_map = {
        "Lisa": "en",
        "Michael": "en",
        "Allison": "en",
        "Robotic": "en"
    }
    lang = voice_map.get(voice_type, "en")
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filename)
        return True
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return False

# ---------------- LOGIN STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "password" not in st.session_state:
    st.session_state.password = ""

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.username = username.strip()
            st.session_state.password = password
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Please enter both username and password.")
else:
    st.title("📖 AI Audiobook & Tone Rewriter")
    st.caption("History saved per user+password combination in /history")

    if "history" not in st.session_state:
        st.session_state.history = load_history(st.session_state.username, st.session_state.password)

    tone = st.selectbox("Choose tone:", ["Neutral", "Inspiring", "Suspenseful", "Exciting", "Romantic", "Funny"])
    voice_type = st.selectbox("Choose voice type:", ["Lisa", "Michael", "Allison", "Robotic"])
    user_text = st.text_area("Enter your text:", height=180, placeholder="Paste or type text here...")
    generate = st.button("Generate Audiobook")

    if generate and user_text.strip():
        ts = int(time.time())

        # Rewrite text
        with st.spinner(f"Rewriting text using {model_type}..."):
            rewritten = rewrite_text(user_text, tone)

        # Generate audio in a temporary file
        with st.spinner("Generating audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                audio_path = tmp_file.name
            audio_ok = generate_audio(rewritten, voice_type, audio_path)

        # Side-by-side original vs rewritten
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.write(user_text)
        with c2:
            st.markdown(f"**Rewritten ({tone})**")
            st.write(rewritten)

        # Play + download audio
        if audio_ok and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download MP3",
                data=audio_bytes,
                file_name=f"audiobook_{ts}.mp3",
                mime="audio/mpeg",
                key=f"download_gen_{ts}"
            )

        # Save history
        entry = {
            "timestamp": ts,
            "tone": tone,
            "voice": voice_type,
            "original": user_text,
            "rewritten": rewritten,
            "audio_path": audio_path if audio_ok else None
        }
        st.session_state.history.append(entry)
        save_history(st.session_state.username, st.session_state.password, st.session_state.history)

    # ---------------- History ----------------
    if st.session_state.history:
        st.subheader("📜 History")
        for i, item in enumerate(reversed(st.session_state.history), start=1):
            title = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item["timestamp"]))
            tone_display = item.get("tone", "Unknown Tone")
            voice_display = item.get("voice", "Unknown Voice")
            with st.expander(f"{i}. {title} — {tone_display} ({voice_display})"):
                st.markdown("**Original**")
                st.write(item["original"])
                st.markdown("**Rewritten**")
                st.write(item["rewritten"])
                if item.get("audio_path") and os.path.exists(item["audio_path"]):
                    with open(item["audio_path"], "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        "⬇️ Download MP3",
                        data=audio_bytes,
                        file_name=os.path.basename(item["audio_path"]),
                        mime="audio/mpeg",
                        key=f"download_hist_{item['timestamp']}_{i}"
                    )

    st.divider()
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
