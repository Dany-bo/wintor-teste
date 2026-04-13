import ollama
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# ==================================================
# 1. CARREGAR MÓDULOS DE ROTINAS (modulo_*.json)
# ==================================================
documentos_rotinas = []
modulos_carregados = []
rotinas_dict = {}  # Dicionário para busca rápida por código

for arquivo in os.listdir("."):
    if arquivo.startswith("modulo_") and arquivo.endswith(".json"):
        try:
            with open(arquivo, "r", encoding='utf-8') as f:
                modulo = json.load(f)
            dados_modulo = modulo["modulo"]
            nome_modulo = dados_modulo["nome"]
            rotinas = dados_modulo["rotinas"]
            for rotina in rotinas:
                codigo = rotina["codigo"]
                nome = rotina["nome"]
                explicacao = rotina.get("explicacao", "Explicação não informada.")
                proxima = rotina.get("proxima_rotina_sugerida", "Nenhuma")
                dependencias = rotina.get("dependencias", [])
                dep_texto = ", ".join(map(str, dependencias)) if dependencias else "Nenhuma"
                
                rotinas_dict[codigo] = {
                    "nome": nome,
                    "explicacao": explicacao,
                    "proxima": proxima,
                    "dependencias": dep_texto
                }
                
                texto = f"""
Módulo: {nome_modulo}
Rotina: {codigo} - {nome}
Explicação: {explicacao}
Próxima rotina sugerida: {proxima}
Dependências: {dep_texto}
"""
                documentos_rotinas.append(Document(page_content=texto))
            modulos_carregados.append(nome_modulo)
            print(f"✅ Módulo '{nome_modulo}' carregado com {len(rotinas)} rotinas.")
        except Exception as e:
            print(f"❌ Erro ao ler {arquivo}: {e}")

if not modulos_carregados:
    print("⚠️ Nenhum módulo encontrado. Coloque arquivos 'modulo_*.json' na pasta.")
    exit()

# ==================================================
# 2. CRIAR ÍNDICE VETORIAL (para buscas semânticas)
# ==================================================
print(f"\n🔍 Criando índice com {len(documentos_rotinas)} documentos...")
embeddings = OllamaEmbeddings(model="llama3.1")
banco = Chroma.from_documents(documentos_rotinas, embeddings)
print("✅ Índice criado!\n")

# ==================================================
# 3. FUNÇÃO DE CONSULTA (sempre usando IA)
# ==================================================
def perguntar(entrada_usuario):
    entrada = entrada_usuario.strip()
    
    # --- Se for apenas número, busca no dicionário e monta contexto ---
    if entrada.isdigit():
        codigo = int(entrada)
        if codigo in rotinas_dict:
            info = rotinas_dict[codigo]
            contexto = f"""
Rotina {codigo} - {info['nome']}
Explicação: {info['explicacao']}
Próxima rotina sugerida: {info['proxima']}
Dependências: {info['dependencias']}
"""
            prompt = f"""Você é um assistente especialista no sistema WinThor.
O usuário perguntou sobre a rotina {codigo}.
Com base nas informações abaixo, responda de forma natural, explicativa e direta, como se estivesse conversando com um usuário. Não faça "copia e cola" do texto; use suas palavras, mas mantenha a precisão.

INFORMAÇÕES:
{contexto}

RESPOSTA:"""
            resposta = ollama.generate(model='llama3.1', prompt=prompt)
            return resposta['response']
        else:
            return f"❌ Não encontrei a rotina {codigo}. Verifique o código."

    # --- Caso contrário: busca semântica normal ---
    resultados = banco.similarity_search(entrada, k=2)
    if not resultados:
        return "❌ Não entendi. Tente perguntar de outra forma ou informe o número da rotina (ex: 210)."
    
    contexto = "\n\n".join([doc.page_content for doc in resultados])
    
    prompt = f"""Você é assistente especialista no sistema WinThor.
Responda de forma clara, direta e natural, usando APENAS as informações abaixo.

CONTEXTO:
{contexto}

PERGUNTA DO USUÁRIO: {entrada}

RESPOSTA:"""
    
    resposta_ollama = ollama.generate(model='llama3.1', prompt=prompt)
    return resposta_ollama['response']

# ==================================================
# 4. INTERFACE DE CONVERSA
# ==================================================
print("=" * 60)
print(" Assistente WinThor - IA Local")
print(f" Módulos disponíveis: {', '.join(modulos_carregados)}")
print("=" * 60)
print("Comandos:")
print("  - Digite o número da rotina (ex: 210) para obter explicação")
print("  - Faça perguntas em português (ex: 'como cadastrar fornecedor')")
print("  - Digite 'sair' para encerrar")
print("=" * 60)

while True:
    entrada = input("\n👤 Você: ").strip()
    if entrada.lower() == "sair":
        print("👋 Até mais!")
        break
    
    print("🤔 Processando...")
    resposta = perguntar(entrada)
    print(f"\n🤖 Assistente WinThor:\n{resposta}")