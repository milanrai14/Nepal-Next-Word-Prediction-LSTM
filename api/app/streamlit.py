import streamlit as st
import requests



st.set_page_config(
    page_title="Nepali Text Generator",
    page_icon="✍️"
)




st.title("✍️ About Nepal LSTM Text Generator")

st.write(
    "Enter some text and generate up to 20 words."
)


seed_text = st.text_area(
    "Enter starting text",
    placeholder="Enter the words"
)


num_words = st.slider(
    "Number of words",
    min_value=1,
    max_value=20,
    value=20
)




if st.button("Generate Text"):

    if not seed_text.strip():

        st.warning("Please enter some text.")

    else:

        payload = {
            "text": seed_text,
            "num_words": num_words
        }

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload
            )

            if response.status_code == 200:

                result = response.json()

                st.subheader("Generated Text")

                st.write(
                    result["generated_text"]
                )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Please start the API first."
            )