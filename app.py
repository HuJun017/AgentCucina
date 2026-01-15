import os
import json
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

load_dotenv()

app = Flask(__name__)
app.secret_key = "kitchen_agent_manual"

# 1. SETUP LLM & TOOLS
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Definiamo il tool di ricerca
tavily_tool = TavilySearchResults(k=3)

def get_recipes_search(query):
    """Esegue la ricerca web e restituisce i risultati"""
    return tavily_tool.invoke({"query": query})

# 2. PROMPT DI SISTEMA
SYSTEM_PROMPT = """Sei un Agente Culinario Metodico e Analitico. Il tuo compito è minimizzare l'incertezza e massimizzare la coerenza. 

--- PROTOCOLLO DI NON-ASSUNZIONE (Rigido) ---
1. MAI assumere il pasto (pranzo/cena/spuntino). Se non lo sai, chiedi.
2. MAI assumere l'abilità dell'utente. Chiedi esplicitamente: "Qual è il tuo livello in cucina? (Principiante, Intermedio, Esperto)".
3. COERENZA INGREDIENTI: Se l'utente elenca un ingrediente (es. pesce), DEVI usarlo o spiegare perché non puoi (es. per un'allergia dichiarata). Non ignorare mai un dato fornito.
4. GESTIONE VINCOLI: Le allergie sono filtri critici. Se c'è un'allergia al pesce ma hai del pesce in frigo, chiedi: "Ho notato che hai del pesce ma c'è un'allergia; vuoi che lo cucini solo per chi può mangiarlo o lo scartiamo del tutto?".

--- FASE DI INTERROGAZIONE (Checklist Obbligatoria) ---
Prima di proporre QUALSIASI ricetta, devi avere conferma di:
- [ ] Lista Ingredienti completa (Tipo, Quantità, Scadenza per i freschi).
- [ ] Numero esatto di persone.
- [ ] Vincoli Salutari (Allergie/Intolleranze).
- [ ] Livello di abilità culinaria (Chiedi se conoscono tecniche specifiche se intendi proporle).
- [ ] Occasione e Tempo a disposizione.

--- REGOLE DI RAGIONAMENTO ---
- Se ricevi vincoli multipli (es. 4 persone + allergia), riepilogali prima di procedere per assicurarti di aver capito.
- Se le informazioni sono in conflitto, FERMATI e chiedi chiarimenti. Non tirare a indovinare.

--- FORMATO JSON ---
{
    "pensiero": "Analisi logica: ho ignorato il pesce perché è emersa un'allergia? Ho chiesto il livello di abilità?",
    "sidebar_data": {
        "ingredienti": [{"tipo": "...", "quantita": "...", "scadenza": "..."}],
        "preferenze": [],
        "vincoli": [],
        "contesto": {"persone": "?", "occasione": "?", "livello_utente": "?"}
    },
    "messaggio_chat": "Markdown test...",
    "bisogno_ricerca": false,
    "query_ricerca": "..."
}
"""

@app.route('/')
def index():
    session['chat_history'] = []
    session['inventory'] = {
        "ingredienti": [], 
        "preferenze": [], 
        "vincoli": [], 
        "contesto": {"persone": "?", "occasione": "non specificata"} # Inizializzazione
    }
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    chat_history = session.get('chat_history', [])
    current_inv = session.get('inventory', {})

    # Prepariamo i messaggi per il modello
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Aggiungiamo la cronologia
    for role, content in chat_history:
        if role == "human": messages.append(HumanMessage(content=content))
        else: messages.append(AIMessage(content=content))
    
    # Aggiungiamo l'input attuale con lo stato della sidebar
    messages.append(HumanMessage(content=f"INPUT: {user_input} | STATO ATTUALE: {json.dumps(current_inv)}"))

    try:
        # PRIMO PASSO: Il modello ragiona e decide se serve una ricerca
        response = llm.invoke(messages)
        content = response.content
        
        # Pulizia JSON
        start = content.find('{')
        end = content.rfind('}') + 1
        data = json.loads(content[start:end])

        # SECONDO PASSO: Se l'agente ha deciso che serve una ricerca (Tool Manuale)
        if data.get("bisogno_ricerca") and data.get("query_ricerca"):
            risultati_web = get_recipes_search(data["query_ricerca"])
            
            # Chiediamo al modello di integrare i risultati della ricerca nella risposta finale
            messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=f"RISULTATI RICERCA WEB: {risultati_web}. Ora proponi le 3 ricette complete basandoti su questi."))
            
            final_response = llm.invoke(messages)
            final_content = final_response.content
            
            start = final_content.find('{')
            end = final_content.rfind('}') + 1
            data = json.loads(final_content[start:end])

        # Aggiorna sessione
        session['inventory'] = data['sidebar_data']
        chat_history.append(("human", user_input))
        chat_history.append(("ai", data['messaggio_chat']))
        session['chat_history'] = chat_history

        return jsonify(data)

    except Exception as e:
        print(f"ERRORE: {e}")
        return jsonify({"messaggio_chat": "C'è stato un errore nel ragionamento. Prova a riformulare.", "sidebar_data": current_inv})

if __name__ == '__main__':
    app.run(debug=True, port=5000)