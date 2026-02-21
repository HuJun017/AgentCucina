import os
import json
import re
import logging
from flask import Flask, render_template, request, Response
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback")

# 1. INIZIALIZZAZIONE MODELLO E TOOL
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

TRUSTED_RECIPE_DOMAINS = [
    "giallozafferano.it",
    "cucchiaio.it",
    "cookaround.com",
    "ricette.it",
    "lacucinaitaliana.it",
    "fattoincasadabenedetta.it",
    "dissapore.com",
    "bbcgoodfood.com",
    "allrecipes.com",
    "seriouseats.com",
]

tavily_tool = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_domains=TRUSTED_RECIPE_DOMAINS,
)

MAX_HISTORY_TURNS = 10

EMPTY_INVENTORY = {
    "ingredienti": [], "preferenze": [], "vincoli": [],
    "contesto": {"persone": "?", "occasione": "?", "livello_utente": "?"}
}

def execute_search_tool(query):
    """Esegue materialmente la ricerca su internet tramite Tavily."""
    try:
        logger.info("Chiamata tool Tavily per query: %s", query)
        return tavily_tool.invoke({"query": query})
    except Exception as e:
        return f"Errore durante la ricerca: {e}"

# 2. PROMPT DI SISTEMA
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
   CRITICO: NON mandare mai un messaggio intermedio del tipo "Grazie, ora cerco..." con 'bisogno_ricerca': false. Se devi cercare, impostalo SUBITO a true nello stesso turno. L'utente non deve inviare un altro messaggio per sbloccarti.

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
    return render_template('index.html')

def extract_json(text):
    """Estrae il JSON gestendo caratteri di controllo non validi."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError:
            clean_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
            return json.loads(clean_str, strict=False)
    raise ValueError("JSON non trovato")

def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

def _build_messages(chat_history, current_inv, user_input):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, content in chat_history:
        messages.append(HumanMessage(content=content) if role == "human" else AIMessage(content=content))
    messages.append(HumanMessage(content=(
        f"INPUT UTENTE: {user_input}\n"
        f"STATO ATTUALE SIDEBAR: {json.dumps(current_inv)}\n\n"
        "ISTRUZIONE: Aggiorna la sidebar includendo i nuovi dati E MANTENENDO quelli vecchi. "
        "NON usare mai la parola \"string\" come valore. Usa \"?\" se non sai qualcosa."
    )))
    return messages

@app.route('/chat', methods=['POST'])
def chat():
    body = request.json or {}
    user_input = body.get("message", "").strip()
    # Lo stato arriva dal client (stateless server-side)
    chat_history = body.get("chat_history", [])
    current_inv = body.get("sidebar_data", EMPTY_INVENTORY)

    def generate():
        yield _sse_event({"type": "status", "msg": "Sto analizzando la tua richiesta..."})
        messages = _build_messages(chat_history, current_inv, user_input)
        try:
            response = llm.invoke(messages)
            data = extract_json(response.content)
            logger.info("Prima LLM call ok. bisogno_ricerca=%s", data.get("bisogno_ricerca"))

            if data.get("bisogno_ricerca") and data.get("query_ricerca"):
                yield _sse_event({"type": "status", "msg": "Cerco ricette online..."})
                search_results = execute_search_tool(data["query_ricerca"])
                logger.info("Tavily ok per query: %s | risultati: %s", data["query_ricerca"],
                    [r.get("url") for r in search_results.get("results", [])] if isinstance(search_results, dict) else "raw")
                messages.append(AIMessage(content=response.content))
                messages.append(HumanMessage(content=f"RISULTATI REALI DAL WEB: {search_results}. Ora genera la risposta finale includendo i link reali."))
                yield _sse_event({"type": "status", "msg": "Preparo la risposta finale..."})
                final_response = llm.invoke(messages)
                data = extract_json(final_response.content)
                logger.info("Seconda LLM call ok.")

            new_history = chat_history + [["human", user_input], ["ai", data["messaggio_chat"]]]
            if len(new_history) > MAX_HISTORY_TURNS * 2:
                new_history = new_history[-(MAX_HISTORY_TURNS * 2):]

            yield _sse_event({"type": "done", "data": data, "chat_history": new_history})

        except Exception as e:
            logger.error("Errore agente: %s", e, exc_info=True)
            yield _sse_event({"type": "error", "data": {"messaggio_chat": "Mi sono perso tra i sapori! Puoi ripetere?", "sidebar_data": current_inv}})

    resp = Response(generate(), mimetype='text/event-stream')
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Cache-Control"] = "no-cache"
    return resp

if __name__ == '__main__':
    app.run(debug=True, port=5000)
