# 🕵️ Investigador IA – Projeto de Inteligência Artificial

Este projeto tem como objetivo aplicar técnicas de IA para responder perguntas baseadas em documentos investigativos, simulando um assistente inteligente capaz de analisar e compreender informações textuais com precisão e contexto.

## 🎯 Objetivo

Criar uma ferramenta capaz de auxiliar em análises investigativas por meio de um sistema de perguntas e respostas baseado em recuperação de documentos e geração de texto com IA.

---

## 🚀 Tecnologias Utilizadas

- **🔎 LangChain** –  Framework de código aberto projetado para facilitar o desenvolvimento de aplicações
- **📚 Transformers** – Para utilizar o modelo de linguagem Flan-T5.
- **🧠 Modelo IA**: `google/flan-t5-base` leve e ideal para CPU.
- **🔤 Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **📁 Vetorstore**: Chroma (armazenamento e busca por similaridade).
- Pipeline:Geração de texto.
- **🌐 Interface**: Streamlit (web app simples e funcional).

---

## 🛠️ Como Funciona

1. O usuário faz uma pergunta.
2. O sistema busca os documentos mais relevantes (via Chroma).
3. Os documentos são passados para o modelo de linguagem Flan-T5.
4. O modelo gera uma resposta com base nos dados encontrados.

---

## 📂 Estrutura do Projeto

```bash
📁 investigadora-ia/
├── app.py               # Interface principal em Streamlit
├── ingest.py            # Código para leitura e vetorização dos documentos
├── documents/           # Pasta com arquivos .txt (documentos investigativos)
├── chroma/              # Banco de vetores gerado (Chroma DB)
├── requirements.txt     # Dependências do projeto
└── README.md           
