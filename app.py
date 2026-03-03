import os
import json
import logging
from flask import Flask, render_template, request, Response
from dotenv import load_dotenv
from groq import Groq
from langchain_tavily import TavilySearch

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tavily = TavilySearch(
    max_results=3,
    search_depth="basic",
    include_domains=[
        "giallozafferano.it",
        "cucchiaio.it",
        "cookaround.com",
        "bbcgoodfood.com",
        "allrecipes.com",
        "seriouseats.com",
        "lacucinaitaliana.it",
        "fattoincasadabenedetta.it",
    ],
)

MODEL = "llama-3.3-70b-versatile"
CRITIC_MODEL = "llama-3.1-8b-instant"  # modello leggero per il critico
MAX_HISTORY = 20  # messaggi (coppie user/assistant)
MAX_AGENT_STEPS = 6  # limite iterazioni agent loop per evitare loop infiniti

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": "Cerca ricette su internet. Usa quando hai ingredienti e contesto sufficienti.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_pantry",
            "description": "Salva ingredienti, preferenze, vincoli e contesto dell'utente. Includi sempre tutti i dati precedenti più i nuovi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredienti": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tipo": {"type": "string"},
                                "quantita": {"type": "string"},
                                "scadenza": {"type": "string"},
                            },
                            "required": ["tipo", "quantita", "scadenza"],
                        },
                    },
                    "preferenze": {"type": "array", "items": {"type": "string"}},
                    "vincoli": {"type": "array", "items": {"type": "string"}},
                    "contesto": {
                        "type": "object",
                        "properties": {
                            "persone": {"type": "string"},
                            "occasione": {"type": "string"},
                            "livello_utente": {"type": "string"},
                        },
                        "required": ["persone", "occasione", "livello_utente"],
                    },
                },
                "required": ["ingredienti", "preferenze", "vincoli", "contesto"],
            },
        },
    },
]

SYSTEM_PROMPT = """Sei Chef Marco, chef italiano anti-spreco. Aiuti a cucinare con ciò che si ha, dando priorità agli ingredienti in scadenza.

**Regola:** UN SOLO tool per turno — mai `update_pantry` e `search_recipes` insieme.

**Raccolta dati:** chiedi ingredienti (con scadenza per freschi), persone, allergie, occasione, livello. Max 2 domande per turno. Appena hai nuovi dati → `update_pantry`.

**Ricette:** dopo `update_pantry`, al turno successivo usa `search_recipes`. Proponi 2-3 opzioni con link, tempi, difficoltà e ingredienti extra. Dai priorità assoluta a ciò che scade prima.

**Stile:** cordiale, diretto, markdown per le ricette, un consiglio tecnico utile per ricetta.

**Critico:** ogni risultato di `search_recipes` include una valutazione [CRITICO]. Se dice "sufficienti", non cercare ancora. Se dice "cambia approccio", usa una query diversa o rispondi con ciò che hai."""


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def get_previous_queries(messages: list) -> list[str]:
    """Estrae le query di search_recipes già eseguite dalla chat history."""
    queries = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                if tc.get("function", {}).get("name") == "search_recipes":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        q = args.get("query", "")
                        if q:
                            queries.append(q)
                    except (json.JSONDecodeError, KeyError):
                        pass
    return queries


def run_critic(query: str, results: list, pantry: dict, prev_queries: list) -> dict:
    """
    Chiama un LLM leggero per valutare se i risultati della ricerca sono sufficienti.
    Restituisce {"sufficient": bool, "feedback": str}.
    """
    critic_messages = [
        {
            "role": "system",
            "content": (
                "Sei un critico di ricerche culinarie. Valuta se i risultati sono utili.\n"
                "Rispondi SOLO con JSON valido: {\"sufficient\": true/false, \"feedback\": \"stringa breve\"}\n\n"
                "Regole:\n"
                "- sufficient=true se almeno 1 risultato contiene una ricetta concreta con ingredienti\n"
                "- sufficient=false se i risultati sono vuoti, irrilevanti o quasi identici a ricerche precedenti\n"
                "- Se la query attuale è semanticamente simile a una precedente, scrivi feedback con suggerimento alternativo"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query_attuale": query,
                    "query_precedenti": prev_queries,
                    "risultati": [
                        {"title": r.get("title", ""), "snippet": r.get("snippet", "")[:200]}
                        for r in results[:2]
                    ],
                    "vincoli_utente": pantry.get("vincoli", []),
                },
                ensure_ascii=False,
            ),
        },
    ]

    try:
        resp = groq_client.chat.completions.create(
            model=CRITIC_MODEL,
            messages=critic_messages,
            temperature=0,
            max_tokens=120,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        # Estrai il JSON anche se il modello aggiunge testo extra
        start, end = raw.find("{"), raw.rfind("}") + 1
        verdict = json.loads(raw[start:end]) if start != -1 else {}
        logger.info("Critico [%s]: sufficient=%s | %s", query[:40], verdict.get("sufficient"), verdict.get("feedback", ""))
        return verdict
    except Exception as e:
        logger.warning("Critico fallito (%s), assumo sufficient=True", e)
        return {"sufficient": True, "feedback": "Valutazione non disponibile."}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    body = request.json or {}
    user_input = body.get("message", "").strip()
    chat_history = body.get("chat_history", [])  # lista di {"role": ..., "content": ...}
    pantry = body.get(
        "pantry",
        {
            "ingredienti": [],
            "preferenze": [],
            "vincoli": [],
            "contesto": {"persone": "?", "occasione": "?", "livello_utente": "?"},
        },
    )

    def generate():
        pantry_state = dict(pantry)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(chat_history[-MAX_HISTORY:])
        messages.append({"role": "user", "content": user_input})

        full_text = ""

        try:
            yield sse({"type": "status", "msg": "Sto pensando..."})

            # Agent loop: continua finché non ci sono più tool calls (max MAX_AGENT_STEPS)
            for _step in range(MAX_AGENT_STEPS):
                response = groq_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=2048,
                )
                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Nessun tool call → risposta testuale finale
                if finish_reason != "tool_calls" or not msg.tool_calls:
                    full_text = msg.content or ""
                    break

                # Aggiungi il messaggio assistant con i tool calls
                assistant_msg: dict = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                }
                if msg.content:
                    assistant_msg["content"] = msg.content
                messages.append(assistant_msg)

                # Esegui ogni tool call
                for tc in msg.tool_calls:
                    fn = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.error("Argomenti tool non validi per %s: %s", fn, tc.function.arguments)
                        args = {}

                    if fn == "search_recipes":
                        q = args.get("query", "ricette italiane")
                        yield sse({"type": "status", "msg": f"Cerco: {q[:55]}..."})
                        try:
                            results = tavily.invoke({"query": q})
                            trimmed = [
                                {
                                    "title": r.get("title", ""),
                                    "url": r.get("url", ""),
                                    "snippet": (r.get("content", "") or "")[:300],
                                }
                                for r in (results if isinstance(results, list) else [])
                            ]
                            result_str = json.dumps(trimmed, ensure_ascii=False)
                            logger.info("Tavily OK: %d chars per '%s'", len(result_str), q)

                            # Critico: valuta i risultati e inietta il verdetto
                            yield sse({"type": "status", "msg": "Valuto i risultati..."})
                            prev_queries = get_previous_queries(messages)
                            verdict = run_critic(q, trimmed, pantry_state, prev_queries)
                            if verdict.get("sufficient", True):
                                result_str += f'\n\n[CRITICO]: Risultati sufficienti. Non cercare ulteriormente.'
                            else:
                                fb = verdict.get("feedback", "Risultati non utili.")
                                result_str += f'\n\n[CRITICO]: {fb} Evita query simili alle precedenti: {prev_queries}.'
                        except Exception as e:
                            result_str = f"Errore nella ricerca: {e}"
                            logger.error("Tavily error: %s", e)

                    elif fn == "update_pantry":
                        pantry_state = args
                        yield sse({"type": "pantry", "data": pantry_state})
                        result_str = "Dispensa aggiornata correttamente."
                        logger.info("Pantry: %d ingredienti", len(pantry_state.get("ingredienti", [])))

                    else:
                        result_str = f"Tool '{fn}' non disponibile."
                        logger.warning("Tool sconosciuto: %s", fn)

                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                    )

                yield sse({"type": "status", "msg": "Elaboro la risposta..."})
            else:
                # Raggiunto il limite di passi senza risposta testuale
                logger.warning("Agent loop: raggiunto MAX_AGENT_STEPS (%d)", MAX_AGENT_STEPS)
                full_text = "Ho elaborato tutte le informazioni. Dimmi pure cosa vorresti cucinare!"

            # Aggiorna la chat history e invia l'evento finale
            new_history = chat_history + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": full_text},
            ]
            if len(new_history) > MAX_HISTORY:
                new_history = new_history[-MAX_HISTORY:]

            yield sse(
                {
                    "type": "done",
                    "message": full_text,
                    "pantry": pantry_state,
                    "chat_history": new_history,
                }
            )

        except Exception as e:
            logger.error("Errore agente: %s", e, exc_info=True)
            yield sse({"type": "error", "message": "Mi sono perso tra i fornelli! Puoi ripetere?"})

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


if __name__ == "__main__":
    app.run(debug=True, port=5000)
