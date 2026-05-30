import os
import time
import json
import logging
import requests
import schedule
from datetime import datetime, timezone, timedelta
from openai import OpenAI

# ================= CONFIGURACIÓN =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("tipster.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Variables de entorno
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([OPENROUTER_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID]):
    raise ValueError("❌ Faltan variables de entorno: OPENROUTER_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID")

# Cliente OpenRouter con modelo que tiene acceso a internet
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://railway.app",
        "X-Title": "Tipster Deportivo IA"
    }
)
MODELO = "google/gemini-3.1-flash-image-preview:online"   # con capacidad de navegación web

FUENTES_CONFIABLES = """
- www.espn.com (estadísticas, resultados, lesiones)
- www.flashscore.com (resultados en vivo, enfrentamientos directos)
- www.sports-reference.com (estadísticas históricas)
- www.understat.com (xG y estadísticas avanzadas de fútbol)
- www.basketball-reference.com (NBA y baloncesto)
- www.mlb.com (béisbol: calendarios, estadísticas, lesiones)
- www.fangraphs.com (estadísticas avanzadas de béisbol)
"""

# ================= TELEGRAM =================
def telegram_mensaje(texto, parse_mode="HTML"):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": texto,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        requests.post(url, data=payload, timeout=15)
    except Exception as e:
        logger.error(f"Error Telegram: {e}")

# ================= BÚSQUEDA DE PARTIDOS (6 POR DEPORTE) =================
def buscar_partidos_destacados():
    """
    Pide a la IA que busque en internet al menos 6 partidos de fútbol,
    6 de baloncesto y 6 de béisbol para el día siguiente.
    """
    manana = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%d')
    prompt = f"""
    Eres un tipster profesional con acceso a internet. Necesito que busques en internet los partidos más importantes que se jugarán el día {manana} (mañana) en los siguientes deportes:

    - **Fútbol** (ligas: Champions, Europa League, Premier, LaLiga, Serie A, Bundesliga, Ligue 1, Libertadores, Brasileirão, etc.)
    - **Baloncesto** (NBA, Euroliga, ACB, Liga Endesa, etc.)
    - **Béisbol** (MLB, Liga Mexicana, ligas invernales, etc.)

    Para CADA deporte, selecciona al menos **6 partidos/eventos** (si hay más de 6 destacados, puedes incluir más). Debes priorizar:
    - Partidos de alta relevancia (playoffs, clásicos, eliminatorias directas)
    - Rivalidades históricas
    - Momento de forma actual
    - Interés mediático y de apuestas

    Para cada partido, proporciona:
    - deporte ("fútbol", "baloncesto" o "béisbol")
    - liga/torneo (ej. "MLB", "NBA", "Champions League")
    - encuentro (ej. "Yankees vs Red Sox", "Lakers vs Celtics")
    - fecha_hora_utc (si la encuentras, en formato "YYYY-MM-DD HH:MM UTC", si no, pon "Horario por confirmar")
    - destacado_razon (breve, máximo 30 palabras)

    Responde ÚNICAMENTE con un JSON válido, sin texto adicional, con esta estructura:

    {{
      "partidos": [
        {{ "deporte": "fútbol", "liga": "...", "encuentro": "...", "fecha_hora_utc": "...", "destacado_razon": "..." }},
        ... (al menos 6 de fútbol)
        {{ "deporte": "baloncesto", ... }}, ... (al menos 6)
        {{ "deporte": "béisbol", ... }}, ... (al menos 6)
      ]
    }}

    Asegúrate de que los eventos sean REALES y estén programados para mañana. Usa tu capacidad de navegación web para verificarlo.
    Si algún deporte no tiene suficientes partidos relevantes, incluye los que haya y nota que son pocos.
    """
    try:
        response = client.chat.completions.create(
            model=MODELO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2500
        )
        content = response.choices[0].message.content
        content = content.strip().strip('```json').strip('```').strip()
        data = json.loads(content)
        partidos = data.get("partidos", [])
        logger.info(f"Se encontraron {len(partidos)} partidos en total.")
        return partidos
    except Exception as e:
        logger.error(f"Error en buscar_partidos_destacados: {e}")
        return []

# ================= ANÁLISIS POR DEPORTE =================
def analizar_partido(partido):
    deporte = partido.get("deporte", "").lower()
    encuentro = partido.get("encuentro", "")
    liga = partido.get("liga", "")
    
    # Prompt base según deporte
    if deporte == "fútbol":
        prompt_detalle = """
        - Predicción principal: resultado exacto o doble oportunidad.
        - ¿Más o menos de 2.5 goles? (justifica)
        - Total de tarjetas (amarillas+rojas) esperado (ej. "Entre 4 y 6")
        - Total de córners esperado (ej. "Entre 8 y 10")
        - ¿Ambos equipos marcan? (Sí/No)
        - Jugador clave (ej. "Mbappé: más de 0.5 goles")
        """
    elif deporte == "baloncesto":
        prompt_detalle = """
        - Total de puntos del partido (más/menos de) con número (ej. "Más de 215.5")
        - Hándicap favorito (ej. "Celtics -5.5")
        - Jugador estrella y su línea de puntos (ej. "Tatum más de 28.5 pts")
        - ¿Quién ganará el rebote total? (equipo)
        """
    elif deporte == "béisbol":
        prompt_detalle = """
        - Ganador del partido (moneyline)
        - Total de carreras (más/menos de) (ej. "Más de 8.5")
        - Hándicap de carreras (run line) (ej. "Yankees -1.5")
        - Lanzador abridor clave y su efectividad esperada
        - ¿Habrá jonrón? (Sí/No, y qué jugador tiene más probabilidad)
        """
    else:
        prompt_detalle = "- Predicción principal y mercados relevantes."
    
    prompt = f"""
    Actúa como un tipster profesional con acceso a internet. Analiza el siguiente evento:

    Deporte: {deporte}
    Liga: {liga}
    Encuentro: {encuentro}
    Fecha (mañana): { (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%d') }

    Busca información actualizada (estadísticas recientes, lesiones, enfrentamientos directos, clima si aplica, etc.) y realiza las predicciones específicas:

    {prompt_detalle}

    Asigna un **nivel de confianza** del 0 al 100% para la predicción principal, justificándolo con datos concretos (ej. "últimos 5 partidos: 4 veces más de 2.5 goles", "lanzador con ERA 1.20 en últimos 3 salidas").

    Escribe un análisis completo de 120-200 palabras explicando el razonamiento, citando estadísticas clave.

    Responde SOLO con JSON válido:
    {{
      "predicciones": {{
        "prediccion_principal": "texto claro",
        "detalles_mercados": {{
          "mercado1": "valor",
          "mercado2": "valor"
        }},
        "confianza": 85,
        "justificacion_confianza": "breve texto"
      }},
      "analisis_completo": "texto extenso con el análisis detallado...",
      "tips_adicionales": ["tip1", "tip2"]
    }}
    """
    try:
        response = client.chat.completions.create(
            model=MODELO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        content = response.choices[0].message.content
        content = content.strip().strip('```json').strip('```').strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error analizando {encuentro}: {e}")
        return None

def generar_combinada_sugerida(analisis_resumen):
    """
    Analisis_resumen es una lista de dicts con campos: encuentro, deporte, prediccion_principal, confianza.
    """
    prompt = f"""
    Eres un tipster experto en combinadas. Con base en estos análisis:

    {json.dumps(analisis_resumen, indent=2, ensure_ascii=False)}

    Genera 2 o 3 apuestas combinadas (múltiples) uniendo las selecciones con mayor confianza y que no estén correlacionadas negativamente.
    Para cada combinada, incluye:
    - Nombre descriptivo
    - Lista de selecciones (texto claro)
    - Confianza global (0-100%)
    - Razonamiento breve

    Responde SOLO con JSON:
    {{
      "combinadas": [
        {{
          "nombre": "Combinada MLB + NBA",
          "selecciones": ["selección 1", "selección 2", "selección 3"],
          "confianza_global": 70,
          "razonamiento": "texto"
        }}
      ]
    }}
    """
    try:
        response = client.chat.completions.create(
            model=MODELO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200
        )
        content = response.choices[0].message.content
        content = content.strip().strip('```json').strip('```').strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error generando combinada: {e}")
        return {"combinadas": []}

# ================= FORMATEO Y ENVÍO A TELEGRAM =================
def formatear_mensaje_partido(partido, analisis):
    deporte = partido.get("deporte", "").capitalize()
    encuentro = partido.get("encuentro", "Desconocido")
    liga = partido.get("liga", "")
    fecha = partido.get("fecha_hora_utc", "Horario por confirmar")
    
    predicciones = analisis.get("predicciones", {})
    pred_principal = predicciones.get("prediccion_principal", "N/A")
    confianza = predicciones.get("confianza", 0)
    justificacion_confianza = predicciones.get("justificacion_confianza", "")
    detalles = predicciones.get("detalles_mercados", {})
    analisis_texto = analisis.get("analisis_completo", "Sin análisis.")
    tips = analisis.get("tips_adicionales", [])
    
    msg = f"""
<b>⚽ {deporte} - {liga}</b>
<b>🏟️ {encuentro}</b>
📅 <i>{fecha}</i>

🎯 <b>PRONÓSTICO PRINCIPAL:</b>
<blockquote>{pred_principal}</blockquote>

🔍 <b>Mercados específicos:</b>
"""
    for mercado, valor in detalles.items():
        msg += f"• <b>{mercado}:</b> {valor}\n"
    
    msg += f"""
😎 <b>Confianza:</b> {confianza}%
💬 <i>{justificacion_confianza}</i>

💎 <b>Análisis completo:</b>
{analisis_texto}

"""
    if tips:
        msg += "<b>💡 Tips adicionales:</b>\n"
        for tip in tips:
            msg += f"• {tip}\n"
    msg += "\n" + "="*40 + "\n"
    return msg

def enviar_analisis_completo(partidos_con_analisis, combinadas):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    intro = f"""
🚀 <b>BOT TIPSTER DEPORTIVO - PREDICCIONES PARA MAÑANA</b>
🤖 <i>Análisis generado el {now}</i>

Se analizaron los partidos más destacados de fútbol, baloncesto y béisbol (mínimo 6 por deporte).
    """
    telegram_mensaje(intro, parse_mode="HTML")
    
    # Agrupar por deporte para mejor organización (opcional)
    for item in partidos_con_analisis:
        partido = item["partido"]
        analisis = item["analisis"]
        if analisis:
            msg = formatear_mensaje_partido(partido, analisis)
            telegram_mensaje(msg, parse_mode="HTML")
            time.sleep(0.8)  # evitar flood
    
    if combinadas and combinadas.get("combinadas"):
        msg_combi = "<b>🎲 COMBINADAS SUGERIDAS</b>\n\n"
        for idx, comb in enumerate(combinadas["combinadas"], 1):
            msg_combi += f"<b>🔹 {comb.get('nombre', f'Combinada {idx}')}</b>\n"
            msg_combi += "Selecciones:\n"
            for sel in comb.get("selecciones", []):
                msg_combi += f"  • {sel}\n"
            msg_combi += f"🎯 Confianza global: {comb.get('confianza_global', 0)}%\n"
            msg_combi += f"🧠 Razonamiento: {comb.get('razonamiento', '')}\n\n"
        telegram_mensaje(msg_combi, parse_mode="HTML")
    else:
        telegram_mensaje("No se generaron combinadas suficientes.", parse_mode="HTML")

# ================= CICLO PRINCIPAL =================
def ciclo_tipster():
    logger.info("🔄 Iniciando ciclo tipster deportivo (fútbol, baloncesto, béisbol - 6+ cada uno)")
    telegram_mensaje("🔍 Buscando al menos 6 partidos de fútbol, 6 de baloncesto y 6 de béisbol para mañana...", parse_mode="HTML")
    
    # 1. Obtener partidos (mínimo 18 en total)
    partidos = buscar_partidos_destacados()
    if not partidos:
        logger.warning("No se encontraron partidos.")
        telegram_mensaje("⚠️ No se pudo obtener lista de partidos. La IA no encontró suficientes eventos.", parse_mode="HTML")
        return
    
    # Opcional: verificar que haya al menos 6 por deporte, si no, advertir
    conteo = {"fútbol": 0, "baloncesto": 0, "béisbol": 0}
    for p in partidos:
        d = p.get("deporte", "").lower()
        if d in conteo:
            conteo[d] += 1
    aviso = ""
    for dep, cnt in conteo.items():
        if cnt < 6:
            aviso += f"⚠️ Solo {cnt} partidos de {dep} (se pidieron 6). "
    if aviso:
        telegram_mensaje(aviso, parse_mode="HTML")
    
    logger.info(f"Partidos obtenidos: Fútbol={conteo['fútbol']}, Baloncesto={conteo['baloncesto']}, Béisbol={conteo['béisbol']}")
    
    # 2. Analizar cada partido
    analisis_resultados = []
    for partido in partidos:
        logger.info(f"Analizando: {partido.get('encuentro')} ({partido.get('deporte')})")
        analisis = analizar_partido(partido)
        if analisis:
            analisis_resultados.append({"partido": partido, "analisis": analisis})
        else:
            logger.error(f"Falló análisis para {partido.get('encuentro')}")
        time.sleep(2)  # pausa entre llamadas a la API
    
    if not analisis_resultados:
        logger.error("No se pudo analizar ningún partido.")
        telegram_mensaje("❌ Error crítico: ningún análisis válido obtenido.", parse_mode="HTML")
        return
    
    # 3. Preparar resumen para combinadas
    resumen_combinada = []
    for item in analisis_resultados:
        p = item["partido"]
        a = item["analisis"]
        resumen_combinada.append({
            "encuentro": p.get("encuentro"),
            "deporte": p.get("deporte"),
            "prediccion_principal": a.get("predicciones", {}).get("prediccion_principal"),
            "confianza": a.get("predicciones", {}).get("confianza", 0)
        })
    combinadas = generar_combinada_sugerida(resumen_combinada)
    
    # 4. Enviar a Telegram
    enviar_analisis_completo(analisis_resultados, combinadas)
    logger.info("✅ Ciclo completado y enviado.")

# ================= MAIN CON SCHEDULE =================
def main():
    logger.info("🚀 Bot Tipster Deportivo (fútbol, baloncesto, béisbol) iniciado")
    telegram_mensaje("🚀 Bot activo. Se ejecutará diariamente a las 21:00 UTC (18:00 ART/Brasil) y buscará 6+ partidos de cada deporte.", parse_mode="HTML")
    
    # Ejecutar una prueba inmediata (comentar si no se desea)
    ciclo_tipster()
    
    # Programar cada día a las 21:00 UTC
    schedule.every().day.at("21:00").do(ciclo_tipster)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
