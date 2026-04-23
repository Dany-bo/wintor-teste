import ollama
import json
import os
import re
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import easyocr
from flask import Flask, request, jsonify, render_template_string
import traceback
 
# ==================================================
# INICIALIZAÇÃO DO EASYOCR (português e inglês)
# ==================================================
ocr_reader = easyocr.Reader(['pt', 'en'], gpu=False)
 
# ==================================================
# CONFIGURAÇÃO DO MODELO
# ==================================================
MODELO_LLM = "qwen3.5:27b"   # Modelo atual
 
# ==================================================
# CARREGAR MÓDULOS DE ROTINAS (modulo_*.json)
# ==================================================
rotinas_dict = {}
modulos_carregados = []
 
for arquivo in os.listdir("."):
    if arquivo.lower().startswith("modulo_") and arquivo.endswith(".json"):
        try:
            with open(arquivo, "r", encoding='utf-8') as f:
                modulo = json.load(f)
            dados_modulo = modulo["modulo"]
            nome_modulo = dados_modulo["nome"]
            rotinas = dados_modulo["rotinas"]
            for rotina in rotinas:
                codigo = rotina["codigo"]
                rotinas_dict[codigo] = {
                    "nome": rotina["nome"],
                    "explicacao": rotina.get("explicacao", "Explicação não informada."),
                    "proxima": rotina.get("proxima_rotina_sugerida", "Nenhuma"),
                    "dependencias": rotina.get("dependencias", []),
                    "modulo": nome_modulo
                }
            modulos_carregados.append(nome_modulo)
            print(f"✅ Módulo '{nome_modulo}' carregado com {len(rotinas)} rotinas.")
        except Exception as e:
            print(f"❌ Erro ao ler {arquivo}: {e}")
 
if not rotinas_dict:
    print("⚠️ Nenhuma rotina encontrada. Coloque arquivos 'modulo_*.json' na pasta.")
    exit(1)
 
print(f"\n📚 Total de rotinas carregadas: {len(rotinas_dict)}")
 
# ==================================================
# CARREGAR BASE DE ERROS (erros_wintor.json)
# ==================================================
erros_lista = []
if os.path.exists("erros_wintor.json"):
    try:
        with open("erros_wintor.json", "r", encoding='utf-8') as f:
            erros_lista = json.load(f).get("erros", [])
        print(f"✅ Carregados {len(erros_lista)} erros/soluções.")
    except Exception as e:
        print(f"⚠️ Erro ao carregar erros_wintor.json: {e}")
else:
    print("ℹ️ Arquivo erros_wintor.json não encontrado. Apenas rotinas serão usadas.")
 
# ==================================================
# FUNÇÕES AUXILIARES
# ==================================================
def extrair_numero_rotina(texto):
    padroes = [
        r'\b(?:rotina|código|codigo|numero|número|me dê a|mostre a|qual a|a rotina|fala da|explique a|o que é a|sobre a|consulte a|pesquise a)\s*(\d{3,4})\b',
        r'\b(\d{3,4})\b'
    ]
    for padrao in padroes:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None
 
def rotinas_proximas(codigo, limite=5):
    codigos = sorted(rotinas_dict.keys())
    if codigo in codigos:
        return []
    pos = 0
    for i, c in enumerate(codigos):
        if c > codigo:
            pos = i
            break
    return codigos[max(0, pos-limite//2):pos+limite//2]
 
def extrair_texto_imagem_bytes(imagem_bytes):
    try:
        imagem = Image.open(BytesIO(imagem_bytes))
        imagem_np = np.array(imagem)
        resultado = ocr_reader.readtext(imagem_np, detail=0, paragraph=True)
        texto = " ".join(resultado)
        return texto.strip()
    except Exception as e:
        print(f"Erro OCR: {traceback.format_exc()}")
        return f"Erro ao processar imagem: {e}"
 
def detectar_erro(texto):
    """Verifica se a pergunta corresponde a algum erro conhecido (por palavras-chave)."""
    texto_lower = texto.lower()
    for erro in erros_lista:
        for palavra in erro.get("palavras_chave", []):
            if palavra.lower() in texto_lower:
                return erro
    return None
 
def responder(pergunta_usuario):
    if not pergunta_usuario:
        return "Digite uma pergunta ou número de rotina."
 
    # 1. PRIORIDADE: detectar erro conhecido
    erro = detectar_erro(pergunta_usuario)
    if erro:
        resposta = f"**🔍 Problema detectado:** {erro.get('titulo', 'Erro desconhecido')}\n\n"
        resposta += f"**Causa:** {erro.get('causa', 'Não informada')}\n\n"
        resposta += f"**Solução:**\n{erro.get('solucao', 'Consulte o manual')}\n\n"
        if erro.get('nota_para_ti'):
            resposta += f"**Nota para TI:** {erro['nota_para_ti']}\n\n"
        if erro.get('proximo_passo_sugerido'):
            resposta += f"**Próximo passo sugerido:** {erro['proximo_passo_sugerido']}"
        return resposta
 
    # 2. Fluxo normal (rotinas) com tratamento de exceções
    try:
        codigo = extrair_numero_rotina(pergunta_usuario)
 
        if codigo and codigo in rotinas_dict:
            info = rotinas_dict[codigo]
            prompt = f"""Você é assistente WinThor. Responda de forma natural usando APENAS os dados abaixo.
 
ROTINA {codigo} - {info['nome']}
Módulo: {info['modulo']}
Explicação: {info['explicacao']}
Próxima rotina sugerida: {info['proxima']}
Dependências: {', '.join(map(str, info['dependencias'])) if info['dependencias'] else 'Nenhuma'}
 
PERGUNTA: {pergunta_usuario}
RESPOSTA:"""
            resposta = ollama.generate(model=MODELO_LLM, prompt=prompt)
            return resposta['response']
        elif codigo:
            proximos = rotinas_proximas(codigo)
            sugestao = f"Rotinas próximas a {codigo}: {', '.join(map(str, proximos[:5]))}." if proximos else "Não há rotinas cadastradas com códigos próximos."
            return f"❌ Rotina {codigo} não encontrada.\n\n{sugestao}\n\nSe você souber o que essa rotina faz, pode adicioná-la ao JSON."
 
        # Busca textual
        termos = set(re.findall(r'\b\w{3,}\b', pergunta_usuario.lower()))
        candidatos = []
        for cod, info in rotinas_dict.items():
            texto_busca = f"{info['nome']} {info['explicacao']}".lower()
            if any(termo in texto_busca for termo in termos):
                candidatos.append((cod, info))
 
        if candidatos:
            contexto = ""
            for cod, info in candidatos[:3]:
                contexto += f"Rotina {cod}: {info['nome']}\n   Explicação: {info['explicacao']}\n\n"
            prompt = f"""Você é assistente WinThor. O usuário perguntou: "{pergunta_usuario}"
 
Abaixo estão rotinas relevantes:
 
{contexto}
 
Responda naturalmente, indicando qual rotina (se alguma) atende à pergunta. Se nenhuma atender, peça para informar o número da rotina."""
            resposta = ollama.generate(model=MODELO_LLM, prompt=prompt)
            return resposta['response']
        else:
            return "Não entendi. Para informações sobre rotinas, informe o número (ex: '2075' ou 'rotina 2075'). Para outras dúvidas, reformule a pergunta."
    except Exception as e:
        print(f"❌ Erro no processamento com Ollama: {e}")
        return f"❌ Erro interno: {str(e)}. Verifique se o modelo '{MODELO_LLM}' está disponível (execute 'ollama list' no terminal)."
 
# ==================================================
# SERVIDOR FLASK COM INTERFACE WEB (MODERNA)
# ==================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
# ==================================================
# HTML_TEMPLATE (use o mesmo que você já tem)
# ==================================================
# Coloque aqui o HTML_TEMPLATE completo que você já possui.
# Se você não tiver, avise que eu reinsiro.
# Para evitar repetição, estou assumindo que a variável HTML_TEMPLATE já está definida.
# Caso contrário, descomente o bloco abaixo com o HTML moderno.
 
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
  <title>WinThor IA · Assistente Técnico Inteligente</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    }
    body { background: #081120; }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 270px 1fr;
      background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.20), transparent 25%),
        radial-gradient(circle at right, rgba(34, 211, 238, 0.10), transparent 20%),
        linear-gradient(135deg, #06101f 0%, #0a1630 35%, #0b2147 70%, #10376b 100%);
      color: white;
    }
    .sidebar {
      border-right: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      backdrop-filter: blur(10px);
      padding: 24px 16px;
    }
    .logoBox {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 30px;
      padding: 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
    }
    .logoIcon {
      width: 44px;
      height: 44px;
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, rgba(56,189,248,0.35), rgba(37,99,235,0.35));
      font-weight: bold;
      font-size: 20px;
    }
    .miniText { font-size: 12px; color: #b8c4db; }
    .logoBox h2 { font-size: 18px; }
    .menu { display: flex; flex-direction: column; gap: 10px; }
    .menuItem {
      border: none;
      background: transparent;
      color: #d5dff3;
      text-align: left;
      padding: 14px 16px;
      border-radius: 14px;
      cursor: pointer;
      transition: 0.3s;
      font-weight: 500;
    }
    .menuItem:hover { background: rgba(255,255,255,0.06); color: white; }
    .menuItem.active {
      background: linear-gradient(90deg, rgba(56,189,248,0.18), rgba(37,99,235,0.14));
      border: 1px solid rgba(125, 211, 252, 0.18);
      color: white;
    }
    .sidebarCard {
      margin-top: 28px;
      padding: 18px;
      border-radius: 22px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
    }
    .sidebarCard h3 { margin: 8px 0 10px; }
    .sidebarCard p { color: #c7d2e6; font-size: 14px; line-height: 1.6; }
    .main { padding: 28px; overflow-y: auto; }
    .hero {
      display: grid;
      grid-template-columns: 1.4fr 0.8fr;
      gap: 24px;
      padding: 28px;
      border-radius: 28px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    }
    .tag {
      display: inline-block;
      margin-bottom: 16px;
      padding: 8px 14px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(34, 211, 238, 0.10);
      border: 1px solid rgba(125, 211, 252, 0.18);
      color: #d9f6ff;
    }
    .heroText h1 { font-size: 42px; line-height: 1.15; margin-bottom: 16px; }
    .heroText p { color: #d3ddf0; font-size:
