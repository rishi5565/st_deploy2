import streamlit as st
import pickle

st.set_page_config(page_title="Spam or Ham Predictor", page_icon=":email:", layout="centered")

# Title and Header
st.title("Spam or Ham Predictor")
st.write("Enter a message and click on the 'Predict' button to check if it's spam or ham.")

# Text Input
input_text = st.text_area("Enter Message Here")

# Model Selection
models = {
    "Naive Bayes": "naive_bayes.pkl",
    "K Nearest Neighbors": "knn.pkl",
    "Support Vector Machine": "svm.pkl",
    "Random Forest": "random_forest.pkl",
}

model_selected = st.selectbox("Select a Model", options=list(models.keys()))

# Prediction
if st.button("Predict"):
    with st.spinner("Predicting..."):
        model_file = models[model_selected]

        count_vectorizer = pickle.load(open("count_vector.pkl", "rb"))
        input_text = count_vectorizer.transform([input_text])
        input_text = input_text.toarray().reshape(-1)

        with open(model_file, "rb") as file:
            model = pickle.load(file)
        prediction = model.predict([input_text])[0]

        if prediction == 'spam':
            st.success("This message is spam.")
        else:
            st.success("This message is ham.")
