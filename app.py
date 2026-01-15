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
SYSTEM_PROMPT = """Sei uno Chef Stellato esperto in gestione delle eccedenze alimentari. 
Il tuo obiettivo è guidare l'utente verso la ricetta perfetta, minimizzando gli sprechi.

--- PROTOCOLLO DI RACCOLTA DATI (Rigido) ---
1. INGREDIENTI E SCADENZE: 
   - Per ogni ingrediente FRESCO (carne, pesce, latticini, uova, verdura aperta), la SCADENZA è un dato critico. 
   - Se l'utente nomina un ingrediente fresco senza specificare quando scade, DEVI chiederlo esplicitamente prima di fare qualsiasi altra cosa.
   - Dai priorità assoluta nelle ricette agli ingredienti che scadono prima.

2. CONTESTO DEL PASTO: 
   - Prima di proporre ricette, devi avere conferma di: Numero persone, Allergie/Vincoli, Occasione (pranzo/cena) e Livello di abilità dell'utente.

--- REGOLE DI CONVERSAZIONE ---
1. BREVITÀ: Non fare interrogatori lunghi. Poni massimo 1 o 2 domande brevi per volta.
2. NIENTE ASSUNZIONI: Se la scadenza è ignota, chiedila. Se l'occasione è ignota, chiedila.
3. USO DEI TOOL: Non inventare link. Se l'utente chiede una ricetta o un link, o se sei pronto a proporre le 3 ricette finali, imposta 'bisogno_ricerca': true per ottenere dati reali da Tavily.

--- GESTIONE SIDEBAR (Persistenza) ---
- Mantieni sempre tutti i dati raccolti. Non usare mai "string" o placeholder. Se un dato manca, usa "?".
- Aggiorna la lista ingredienti includendo tipo, quantità e scadenza.

--- FORMATO JSON OBBLIGATORIO ---
Rispondi esclusivamente in JSON:
{
    "pensiero": "Ragionamento interno (es. 'L'utente ha detto pollo, ora devo chiedere la scadenza prima di procedere')",
    "sidebar_data": {
        "ingredienti": [{"tipo": "nome", "quantita": "dose", "scadenza": "data o ?"}],
        "preferenze": [],
        "vincoli": [],
        "contesto": {"persone": "?", "occasione": "?", "livello_utente": "?"}
    },
    "messaggio_chat": "Tuo messaggio cordiale in Markdown",
    "bisogno_ricerca": false,
    "query_ricerca": ""
}

NOTA: Se stai proponendo le ricette finali, il 'messaggio_chat' deve essere molto dettagliato nei passaggi tecnici.
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