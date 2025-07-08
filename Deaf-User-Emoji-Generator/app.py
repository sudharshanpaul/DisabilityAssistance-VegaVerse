import streamlit as st
from groq import Groq

st.set_page_config(page_title="Text to Emoji for Deaf", page_icon="🧏‍♂️", layout="centered")
st.title("🧏‍♂️ Text to Emoji Generator for Deaf People")

# Input for Groq API Key
api_key = st.text_input("Enter your Groq API Key:", type="password")

# Input for sentence
user_input = st.text_input("Enter a sentence:")

# Proceed only if both are provided
if st.button("Generate Emoji") and api_key and user_input:
    with st.spinner("Generating..."):
        try:
            # Initialize Groq client using provided key
            client = Groq(api_key=api_key)

            # Prompt creation
            prompt = f"Convert the following sentence into a meaningful emoji sequence for deaf communication:\n\n\"{user_input}\""

            # Make request to Groq API (✅ updated model)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mistral-saba-24b"  # ✅ REPLACED with working model
            )

            emoji_output = response.choices[0].message.content.strip()

            # Display result
            st.success("Here's your emoji translation:")
            st.markdown(f"<h2 style='text-align: center;'>{emoji_output}</h2>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

elif st.button("Generate Emoji") and not api_key:
    st.warning("⚠️ Please enter your Groq API key.")
