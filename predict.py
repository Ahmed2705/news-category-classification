import streamlit as st
import joblib
import string


model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


st.set_page_config(
    page_title="🧠 News Category Classifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="auto"
)


st.title("🧠 News Category Classifier")
st.write(
    "Classify news articles into **World**, **Sports**, **Business**, or **Science/Technology** using a trained Logistic Regression model.")


st.sidebar.header("🧩 About")
st.sidebar.info(
    """
    This app uses **TF-IDF + Logistic Regression** trained on the **AG News Dataset**  
    to predict the type of a given news article.
    """
)
st.sidebar.write("---")
st.sidebar.write("👨‍💻 Developed by Ahmed Mostafa")


news_input = st.text_area("🗞️ Enter a news headline or article description:", height=150,
                          placeholder="Type or paste your news text here...")


if st.button("🚀 Classify News"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some text first!")
    else:

        text = news_input.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))


        prediction = model.predict(vectorizer.transform([text]))
        probs = model.predict_proba(vectorizer.transform([text]))[0]
        pred = prediction[0]
        confidence = probs[pred - 1] * 100


        categories = {1: "🌍 World", 2: "🏅 Sports", 3: "💼 Business", 4: "💻 Science/Technology"}
        category_name = categories[pred]


        st.success(f"### ✅ Predicted Category: {category_name}")
        st.progress(confidence / 100)
        st.write(f"**Confidence:** {confidence:.2f}%")


        st.subheader("📊 Class Probabilities:")
        prob_dict = {categories[i + 1]: probs[i] * 100 for i in range(4)}
        st.bar_chart(prob_dict)


st.write("---")
st.caption("Logistic Regression • TF-IDF")
