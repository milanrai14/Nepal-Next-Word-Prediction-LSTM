import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Nepal Next Word Predictor", page_icon="🇳🇵")

st.title("🇳🇵 Nepal Next Word Predictor")
st.caption("LSTM-based next-word generation trained on a Nepal text corpus.")

with st.sidebar:
    st.header("Settings")
    n_words = st.slider("Words to generate", 5, 50, 20)
    st.markdown("---")
    st.markdown("**Try these seeds:**")
    st.code("Nepal")
    st.code("culture")
    st.code("pokhara")
    st.code("quantum physics  # not related")

seed = st.text_input("Enter seed text:", value="Nepal")

if st.button("Predict", type="primary"):
    if not seed.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"text": seed, "n_words": n_words},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                if data["related"]:
                    st.success("✅ Input is related to the Nepal corpus")
                    st.markdown("### Generated text")
                    st.info(data["output"])
                else:
                    st.error("❌ Input is NOT related to the Nepal corpus")
                    st.write(data["output"])

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach API. Make sure FastAPI is running:\n\n"
                    "`uvicorn main:app --reload`"
                )
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Backend: FastAPI • Model: LSTM • Dataset: About Nepal")