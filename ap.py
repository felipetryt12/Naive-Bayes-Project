from sklearn.feature_extraction.text import TfidfVectorizer
import os
import streamlit as st
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

modelo_path = os.path.join(BASE_DIR, 'models', 'Multinomial_bayes_for_review_clasification')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl')


if os.path.exists(modelo_path):
    modelo = joblib.load(modelo_path)
    st.write("Modelo cargado correctamente")
else:
    st.error(f"El archivo del modelo no existe en {modelo_path}")

if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
    st.write("Vectorizador cargado correctamente")
else:
    st.error(f"El archivo del vectorizador no existe en {vectorizer_path}")

st.title("Clasificación de Comentarios: Positivo o Negativo")

st.write("""
         Esta aplicación utiliza un modelo de Machine Learning para clasificar comentarios como positivos o negativos. 
         Es importante que los comentarios estén en inglés.
         """)

comentario_usuario = st.text_area("Escribe o pega tu comentario aquí:")

if st.button("Clasificar"):
    if comentario_usuario:
        try:
            comentario_procesado = [comentario_usuario.strip().lower()]

            comentario_vectorizado = vectorizer.transform(comentario_procesado)

            prediccion = modelo.predict(comentario_vectorizado)
            probabilidad = modelo.predict_proba(comentario_vectorizado)[:, 1]

            if prediccion == 1:
                st.success(f"El comentario tiene una probabilidad de {probabilidad[0]*100:.2f}% de ser POSITIVO.")
            else:
                st.error(f"El comentario tiene una probabilidad de {(1 - probabilidad[0])*100:.2f}% de ser NEGATIVO.")
        except Exception as e:
            st.error(f"Error al clasificar el comentario: {e}")
    else:
        st.warning("Por favor, escribe un comentario para clasificar.")
