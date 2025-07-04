import os
import sys
import types
import streamlit as st

import torch
torch.classes = types.SimpleNamespace()
sys.modules["torch.classes"] = torch.classes
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Investigador IA", layout="centered")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "token"

STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #f8f9fa;
    color: #222;
}
h1 {
    color: #2c3e50;
    text-shadow: 1px 1px 2px #aaa;
    font-weight: 700;
}
p {
    font-size: 1.1rem;
    color: #34495e;
    text-align: center;
}
input[type="text"] {
    padding: 12px 15px;
    border-radius: 8px;
    border: 2px solid #3498db;
    font-size: 1rem;
    outline: none;
    transition: border-color 0.3s ease;
    width: 100%;
}
input[type="text"]:focus {
    border-color: #2980b9;
}
"""

# Embeddings: transformam textos em vetores para buscar documentos relacionados 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
persist_directory = "chroma_db"
vetorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)

# modelo de linguagem baseado em FLAN-T5 para gerar respostas 
model_id = "google/flan-t5-base"

@st.cache_resource  #evita carregar o modelo toda hora
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.3
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm_model()

# chain para recuperar documentos relevantes e responder a pergunta 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vetorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

def main():
    st.markdown(f"<style>{STYLES}</style>", unsafe_allow_html=True)

    st.markdown("<h1>🕵️ Investigador IA</h1>", unsafe_allow_html=True)
    st.markdown("<p>Sua ferramenta de análise de documentos investigativos. Faça perguntas e obtenha insights rápidos.</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    pergunta = st.text_input("❓ Digite sua pergunta sobre o caso:", placeholder="Ex: Qual o depoimento da vítima?")

    response_container = st.container()

    if pergunta:
        with st.spinner("🔍 Pesquisando e analisando documentos..."):
            try:
                # envia a pergunta para a chain que busca documentos e gera resposta
                resposta_completa = qa_chain.invoke({"query": pergunta})
                resposta_texto = resposta_completa.get("result", "").strip()
                source_documents = resposta_completa.get("source_documents", [])

                frases_nulas = [
                    "não sei", "não consigo", "no tengo", "não encontrei", "indisponível",
                    "não há informação", "não sou capaz", "não está nos documentos"
                ]

                with response_container:
                    if not source_documents or not resposta_texto or any(f in resposta_texto.lower() for f in frases_nulas):
                        st.info("⚠️ **Informação Não Encontrada:** Reformule a pergunta ou verifique os documentos.")
                    else:
                        st.subheader("✅ Resposta:")
                        st.write(resposta_texto)

            except Exception as e:
                with response_container:
                    st.error(f"❌ Ocorreu um erro: {e}")
                    st.info("Verifique os arquivos do modelo ou tente novamente.")

if __name__ == "__main__":
    main()
    