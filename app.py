import os
import json
import re
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

app = Flask(__name__)
app.secret_key = "chef_agent_final_secure_v3"

# 1. INIZIALIZZAZIONE MODELLO E TOOL
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Tool di ricerca professionale
tavily_tool = TavilySearchResults(k=3)

def execute_search_tool(query):
    """Esegue materialmente la ricerca su internet tramite Tavily."""
    try:
        print(f"DEBUG: Chiamata tool Tavily per query: {query}")
        return tavily_tool.invoke({"query": query})
    except Exception as e:
        return f"Errore durante la ricerca: {e}"

# 2. PROMPT DI SISTEMA (Ottimizzato per evitare il bug "string")
SYSTEM_PROMPT = """Sei uno Chef Stellato e un Agente AI esperto. 
La tua missione è guidare l'utente alla ricetta perfetta basandoti sul contesto reale.

--- REGOLE SIDEBAR (Dati Persistenti) ---
1. NON USARE MAI la parola "string" come valore nel JSON. 
2. Se un dato è sconosciuto, usa il simbolo "?".
3. PERSISTENZA: Ad ogni risposta, devi includere TUTTI gli ingredienti e i dati raccolti in precedenza. Non cancellare mai i dati della sidebar se non su richiesta esplicita.

--- REGOLE DIALOGO ---
1. BREVITÀ: Poni massimo 1 o 2 domande brevi per volta.
2. CONTESTO: Non proporre ricette finché non hai: Ingredienti, Scadenze, Persone, Allergie, Occasione e Abilità.
3. NO LINK FASULLI: Se l'utente chiede un link o una ricetta specifica, imposta 'bisogno_ricerca': true.

--- FORMATO JSON OBBLIGATORIO ---
Rispondi esclusivamente in JSON. Esempio struttura:
{
    "pensiero": "Ragionamento dello chef",
    "sidebar_data": {
        "ingredienti": [{"tipo": "nome", "quantita": "dose", "scadenza": "data o ?"}],
        "preferenze": [],
        "vincoli": [],
        "contesto": {"persone": "?", "occasione": "?", "livello_utente": "?"}
    },
    "messaggio_chat": "Tuo messaggio in Markdown",
    "bisogno_ricerca": false,
    "query_ricerca": ""
}

NOTA: Per andare a capo nel messaggio_chat usa '\\n'.
"""

@app.route('/')
def index():
    session['chat_history'] = []
    session['inventory'] = {
        "ingredienti": [], "preferenze": [], "vincoli": [], 
        "contesto": {"persone": "?", "occasione": "?", "livello_utente": "?"}
    }
    return render_template('index.html')

def extract_json(text):
    """Estrae il JSON gestendo caratteri di controllo non validi e placeholder errati."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # strict=False permette newline e caratteri di controllo
            data = json.loads(json_str, strict=False)
            return data
        except json.JSONDecodeError:
            # Pulizia manuale estrema
            clean_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
            return json.loads(clean_str, strict=False)
    raise ValueError("JSON non trovato")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    chat_history = session.get('chat_history', [])
    current_inv = session.get('inventory', {})

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, content in chat_history:
        if role == "human": messages.append(HumanMessage(content=content))
        else: messages.append(AIMessage(content=content))
    
    # Messaggio Human potenziato per forzare la persistenza dei dati
    messages.append(HumanMessage(content=f"""
    INPUT UTENTE: {user_input}
    STATO ATTUALE SIDEBAR: {json.dumps(current_inv)}
    
    ISTRUZIONE: Aggiorna la sidebar includendo i nuovi dati E MANTENENDO quelli vecchi. 
    NON usare mai la parola "string" come valore. Usa "?" se non sai qualcosa.
    """))

    try:
        # STEP 1: Generazione risposta
        response = llm.invoke(messages)
        data = extract_json(response.content)

        # STEP 2: Gestione Tool Ricerca
        if data.get("bisogno_ricerca") and data.get("query_ricerca"):
            search_results = execute_search_tool(data["query_ricerca"])
            
            messages.append(AIMessage(content=response.content))
            messages.append(HumanMessage(content=f"RISULTATI REALI DAL WEB: {search_results}. Ora genera la risposta finale includendo i link reali."))
            
            final_response = llm.invoke(messages)
            data = extract_json(final_response.content)

        # Aggiornamento sessione
        session['inventory'] = data['sidebar_data']
        chat_history.append(("human", user_input))
        chat_history.append(("ai", data['messaggio_chat']))
        session['chat_history'] = chat_history

        return jsonify(data)

    except Exception as e:
        print(f"ERRORE AGENTE: {e}")
        return jsonify({
            "messaggio_chat": "Mi sono perso tra i sapori! Puoi ripetere l'ultima informazione?",
            "sidebar_data": current_inv
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)