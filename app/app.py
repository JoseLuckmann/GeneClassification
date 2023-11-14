import streamlit as st
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SequencePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  
        X['words'] = X.apply(lambda x: getKmers(x['sequence']), axis=1)
        X["sequence"] = X.apply(lambda row: " ".join(row["words"]), axis=1)
        return X['sequence']

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
    
modelo_especie = joblib.load("./models/modelo_especie.joblib")
modelo_proteina = joblib.load("./models/modelo_proteina.joblib")
st.title("Análise de Sequência de DNA")


dna_input = st.text_area("Insira a Sequência de DNA", height=150)
botao_predizer = st.button("Predizer")


if botao_predizer:
    if dna_input:        
        especie_pred = modelo_especie.predict(pd.DataFrame({"sequence":[dna_input]}))
        proteina_pred = modelo_proteina.predict(pd.DataFrame({"sequence":[dna_input]}))

        # Exibir resultados
        st.write(f"Espécie Prevista: {especie_pred}")
        st.write(f"Tipo de Proteína: {proteina_pred}")
    else:
        st.error("Por favor, insira uma sequência de DNA válida.")
