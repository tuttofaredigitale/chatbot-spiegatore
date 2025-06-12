# --- Import delle librerie necessarie ---
import os
import gradio as gr

# Le librerie LangChain verranno installate tramite requirements.txt
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Controllo della Chiave API di OpenAI
# Su Hugging Face Spaces, le chiavi API vengono caricate come "Secrets"
# e sono disponibili come variabili d'ambiente.
if "OPENAI_API_KEY" not in os.environ:
    print("ERRORE: La variabile d'ambiente OPENAI_API_KEY non è stata trovata.")
    raise ValueError("OPENAI_API_KEY non trovata. Assicurati di averla impostata nei Secrets dello Space.")

print("--- Inizio Avvio Applicazione Chatbot RAG ---")

# Questa sezione è identica a quella della lezione precedente.
# L'app la eseguirà una sola volta all'avvio.

try:
    print("Caricamento del modello di embedding...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Sul cloud usiamo la CPU
    )

    print("Caricamento dell'indice vettoriale FAISS...")
    # Il percorso deve corrispondere alla struttura caricata su HF Spaces
    vector_store = FAISS.load_local(
        "my_faiss_index", # Nome della cartella dell'indice
        embeddings_model,
        allow_dangerous_deserialization=True
    )

    print("Inizializzazione del retriever...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("Inizializzazione del modello LLM...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Definizione del template del prompt
    template_text = """
    Sei un assistente AI utile e preciso. Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul contesto fornito.
    Se le informazioni per rispondere non sono nel contesto, dì: "Mi dispiace, non ho trovato informazioni sufficienti nei documenti per rispondere.".
    Non inventare risposte.

    Contesto:
    {context}

    Domanda:
    {question}

    Risposta Utile:
    """
    rag_prompt_template = PromptTemplate.from_template(template_text)

    # Funzione di formattazione
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    print("Costruzione della catena RAG...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )
    print("Logica RAG inizializzata con successo.")

except Exception as e:
    print(f"Errore critico durante l'inizializzazione: {e}")
    raise e

# Definizione della funzione di interfaccia per Gradio
def get_chatbot_response(message, history):
    """Funzione che Gradio chiama per ogni input dell'utente."""
    try:
        return rag_chain.invoke(message)
    except Exception as e:
        print(f"Errore durante l'invocazione della catena: {e}")
        return "Oops! Qualcosa è andato storto. Riprova."

# Creazione e lancio dell'interfaccia Gradio
print("Configurazione e avvio dell'interfaccia Gradio...")

# Usiamo gr.Blocks per un maggiore controllo e per poter mostrare il titolo
# all'interno del blocco principale dell'app su Spaces.
with gr.Blocks(theme="soft", title="Assistente RAG") as demo:
    gr.Markdown("# Assistente Virtuale RAG 🤖")
    gr.Markdown("Fai una domanda sulla Knowledge Base (es. procedure per il reset password).")
    
    gr.ChatInterface(
        fn=get_chatbot_response,
        examples=[
            "Come posso resettare la mia password?",
            "Quali sono i requisiti per una nuova password?",
            "A che ora è disponibile l'assistenza clienti?"
        ],
        cache_examples=False
    )

# Lancia l'applicazione. Su HF Spaces, non dovrebbe servire share=True.
# server_name="0.0.0.0" permette all'app di essere accessibile
# dall'esterno del container Docker di HF.
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")