import gradio as gr
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import LlamaIndexTool
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool
import os
from dotenv import load_dotenv
import requests
import arxiv
import requests


load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
groq = os.getenv('GROQ_API_KEY')
tavily_key = os.getenv('tavily')
nvidia = os.getenv('nvidia')

Settings.llm = Groq(model="llama3-groq-70b-8192-tool-use-preview", api_key=groq)
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
def consulta_artigos(titulo: str) -> str:
    """
    Consulta os artigos na base de dados arXiv e retorna resultados formatados.
    """
    busca = arxiv.Search(
        query=titulo,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )

    resultados = []
    for resultado in busca.results():
        resultados.append(f"Título: {resultado.title}\n"
                          f"Resumo: {resultado.summary}\n"
                          f"Categoria: {resultado.primary_category}\n"
                          f"Link: {resultado.entry_id}\n")

    return "\n\n".join(resultados)


ferramenta_artigos = FunctionTool.from_defaults(fn=consulta_artigos, name="consulta_artigos", description="Consulta os artigos na base de dados arXiv e retorna resultados formatados.")
tool = LlamaIndexTool.from_tool(ferramenta_artigos)
print(tavily_key)
tavily_tool = TavilyToolSpec(api_key=tavily_key)
tavily_tools = tavily_tool.to_tool_list()
tools = [LlamaIndexTool.from_tool(t) for t in tavily_tools]


storage_context = StorageContext.from_defaults(
        persist_dir="./artigo"
    )
artigo_index = load_index_from_storage(storage_context)

storage_context = StorageContext.from_defaults(
        persist_dir="./livro"
    )
livro_index = load_index_from_storage(storage_context)


artigo_engine = artigo_index.as_query_engine(similarity_top_k=3)
livro_engine = livro_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=artigo_engine,
        metadata=ToolMetadata(
            name="artigo_engine",
            description=(
                "Fornece informações sobre algoritmos de inteligência artificial nas redes sociais. "
                "Use uma pergunta detalhada em texto simples como entrada para a ferramenta."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=livro_engine,
        metadata=ToolMetadata(
            name="livro_engine",
            description=(
                "Fornece informações sobre avanços e tendências sobre inteligência artificial. "
                "Use uma pergunta detalhada em texto simples como entrada para a ferramenta."
            ),
        ),
    ),
]

llm = LLM(
    model="nvidia_nim/meta/llama-3.3-70b-instruct",
    api_key=nvidia
)

def baixar_pdf_arxiv(link):
    """
    Baixa o PDF de um artigo do arXiv dado o link do artigo.

    Args:
        link (str): O link para o artigo no arXiv.

    Returns:
        str: O caminho do arquivo salvo ou uma mensagem de erro.
    """
    try:
        # Verifica se o link é do arXiv
        if "arxiv.org" not in link:
            return "O link fornecido não é um link válido do arXiv."

        # Extrai o ID do artigo do link
        artigo_id = link.split("/")[-1]

        # Monta o link direto para o PDF
        pdf_url = f"https://arxiv.org/pdf/{artigo_id}.pdf"

        # Faz o download do PDF
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            nome_arquivo = f"artigo_{artigo_id}.pdf"
            with open(nome_arquivo, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return f"PDF salvo como {nome_arquivo}"
        else:
            return f"Erro ao baixar o PDF. Código de status: {response.status_code}"
    except Exception as e:
        return f"Ocorreu um erro: {e}"

ferramenta_baixar = FunctionTool.from_defaults(fn=baixar_pdf_arxiv, name="baixa_artigos", description="Baixa o PDF de um artigo do arXiv dado o link do artigo.")
tool_baixar = LlamaIndexTool.from_tool(ferramenta_baixar)

def pesquisar_artigos(tema):
    # Criação dos agentes com o tema dinâmico
    agent = Agent(
        role='Agente de pesquisa',
        goal='Fornece artigos científicos sobre um assunto de interesse.',
        backstory='Um agente expert em pesquisa científica que possui a habilidade de acessar e baixar artigos no arxiv',
        tools=[tool, tool_baixar],
        llm=llm
    )

    agent_web = Agent(
        role='Agente de pesquisa por documentos na web',
        goal='Fornece artigos científicos encontrados na web sobre um assunto de interesse.',
        backstory='Um agente expert em pesquisa científica que possui a habilidade de buscar artigos na web',
        tools=[*tools],
        llm=llm
    )

    agente_verificacao = Agent(
        role='Agente de pesquisa que verifica documentos',
        goal='Fornece como saída apenas artigos científicos válidos',
        backstory='Um agente expert em pesquisa científica',
        tools=[*tools],
        llm=llm
    )

    manager = Agent(
        role="Gerente do projeto",
        goal="Gerenciar a equipe com eficiência",
        backstory="Gerente de projeto experiente que coordena os esforços da equipe",
        allow_delegation=True,
        llm=llm
    )

    # Criação das tasks usando o tema recebido como parâmetro
    task = Task(
        description=f"Busque artigos científicos no arxiv sobre {tema}.",
        expected_output="5 artigos e seus respectivos links",
        agent=agent
    )

    task1 = Task(
        description=f"Busque artigos científicos sobre {tema}.",
        expected_output="5 artigos e seus respectivos links",
        agent=agent_web
    )

    task2 = Task(
        description="Verifique se os artigos encontrados na web realmente são artigos científicos.",
        expected_output="5 artigos e seus respectivos links",
        agent=agente_verificacao
    )

    crew_hierarquica = Crew(
        agents=[agent, agent_web, agente_verificacao],
        tasks=[task, task1, task2],
        manager_agent=manager,
        process=Process.hierarchical,
        verbose=1,
    )

    result = crew_hierarquica.kickoff()
    return result

# Função para a segunda aba
def pergunta(query):
    agente = ReActAgent.from_tools(
    query_engine_tools,
    verbose=True,
    )
    response = agente.chat(query)

    return response

# Interface Gradio para a primeira aba
pesquisa_interface = gr.Interface(
    fn=pesquisar_artigos,
    inputs=gr.Textbox(label="Digite o tema para pesquisa"),
    outputs=gr.Textbox(label="Resultados"),
    title="👨‍🔬👩‍🔬 Pesquisador de Artigos Científicos",
    description="Digite um tema para encontrar artigos científicos relacionados.",
    theme=gr.themes.Glass()
)

# Interface Gradio para a segunda aba
hello_interface = gr.Interface(
    fn=pergunta,
    inputs=gr.Textbox(label="Faça sua pergunta"),
    outputs=gr.Textbox(label="Saída"),
    title="Perguntas sobre IA nas redes sociais",
    description="Exemplo simples de uma segunda aba."
)

# Combinação das abas em um bloco de abas
with gr.Blocks(theme=gr.themes.Glass()) as app:
    with gr.Tab("Pesquisa de Artigos"):
        pesquisa_interface.render()
    with gr.Tab("Consulta artigos"):
        hello_interface.render()

# Iniciar a aplicação
app.launch()
