#!/usr/bin/env python3
"""
Bot de predicciones deportivas con IA (NVIDIA Nemotron a través de OpenRouter)
Funciona 24/7 en Railway. Analiza eventos en vivo/próximos y envía top predicciones a Telegram.
"""

import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sports_odds_api import SportsGameOdds
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes

# ==================== CONFIGURACIÓN ====================
load_dotenv()

class Config:
    SPORTS_ODDS_API_KEY = os.getenv("SPORTS_ODDS_API_KEY_HEADER")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    AI_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    MAX_PREDICTIONS = int(os.getenv("MAX_PREDICTIONS", 10))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))
    ANALYSIS_INTERVAL_SECONDS = int(os.getenv("ANALYSIS_INTERVAL_SECONDS", 3600))
    SPORTS_LEAGUES = os.getenv("SPORTS_LEAGUES", "football,tennis,basketball,baseball").split(",")
    
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://railway.app")
    OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "Sports Prediction Bot")
    
    @classmethod
    def validate(cls):
        required = [cls.SPORTS_ODDS_API_KEY, cls.OPENROUTER_API_KEY, cls.TELEGRAM_BOT_TOKEN, cls.TELEGRAM_CHAT_ID]
        if not all(required):
            missing = [k for k in ["SPORTS_ODDS_API_KEY","OPENROUTER_API_KEY","TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID"] if not locals().get(k)]
            raise ValueError(f"Faltan variables: {missing}")

# ==================== CLIENTE API DE CUOTAS ====================
class OddsClient:
    def __init__(self):
        self.client = SportsGameOdds(api_key_param=Config.SPORTS_ODDS_API_KEY)
        self.sports_mapping = {
            "football": ["NFL","NCAAF","EPL","UCL","LaLiga","SerieA","Bundesliga","LigaMX","MLS"],
            "tennis": ["ATP","WTA"],
            "basketball": ["NBA","NCAA","EuroLeague","ACB"],
            "baseball": ["MLB","KBO","NPB"]
        }
    
    async def get_live_events(self) -> List[Dict]:
        try:
            loop = asyncio.get_event_loop()
            page = await loop.run_in_executor(None, self.client.events.get, {"status": "live"})
            events = page.data if page and hasattr(page, "data") else []
            return self._filter_relevant(events)
        except Exception as e:
            print(f"❌ Error eventos en vivo: {e}")
            return []
    
    async def get_upcoming_events(self) -> List[Dict]:
        try:
            loop = asyncio.get_event_loop()
            page = await loop.run_in_executor(None, self.client.events.get, {"status": "upcoming"})
            events = page.data if page and hasattr(page, "data") else []
            return self._filter_relevant(events)
        except Exception as e:
            print(f"❌ Error eventos próximos: {e}")
            return []
    
    async def get_event_details(self, event_id: str) -> Optional[Dict]:
        try:
            loop = asyncio.get_event_loop()
            event = await loop.run_in_executor(None, self.client.events.get_by_id, event_id)
            return event.data if event and hasattr(event, "data") else None
        except Exception:
            return None
    
    async def verify_result(self, event_id: str) -> Optional[Dict]:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.client.events.get_result, event_id)
            return result.data if result and hasattr(result, "data") else None
        except Exception:
            return None
    
    def _filter_relevant(self, events):
        relevant_leagues = []
        for sport in Config.SPORTS_LEAGUES:
            relevant_leagues.extend(self.sports_mapping.get(sport.strip(), []))
        filtered = []
        for ev in events:
            league = ev.get("league") if isinstance(ev, dict) else getattr(ev, "league", "")
            if league in relevant_leagues:
                filtered.append(self._normalize(ev))
        return filtered
    
    def _normalize(self, ev):
        if isinstance(ev, dict):
            return {
                "id": ev.get("id"),
                "sport": ev.get("sport"),
                "league": ev.get("league"),
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "start_time": ev.get("start_time"),
                "status": ev.get("status"),
                "odds": ev.get("odds", {}),
                "score": ev.get("score", {})
            }
        else:
            return {
                "id": getattr(ev, "id", None),
                "sport": getattr(ev, "sport", None),
                "league": getattr(ev, "league", None),
                "home_team": getattr(ev, "home_team", None),
                "away_team": getattr(ev, "away_team", None),
                "start_time": getattr(ev, "start_time", None),
                "status": getattr(ev, "status", None),
                "odds": getattr(ev, "odds", {}),
                "score": getattr(ev, "score", {})
            }

# ==================== ANALIZADOR IA (OpenRouter + NVIDIA) ====================
class AIAnalyzer:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=Config.OPENROUTER_API_KEY,
            timeout=30
        )
        self.model = Config.AI_MODEL
        self.threshold = Config.CONFIDENCE_THRESHOLD
        self.headers = {"HTTP-Referer": Config.OPENROUTER_SITE_URL, "X-Title": Config.OPENROUTER_SITE_NAME}
    
    async def analyze(self, event: Dict) -> Optional[Dict]:
        prompt = self._build_prompt(event)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un analista deportivo. Responde SOLO con JSON válido, sin texto adicional."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=4000,
                extra_headers=self.headers
            )
            text = response.choices[0].message.content
            pred = self._parse(text, event)
            print(f"📊 Tokens: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
            return pred
        except Exception as e:
            print(f"❌ Error IA: {e}")
            return None
    
    async def rank(self, predictions: List[Dict]) -> List[Dict]:
        valid = [p for p in predictions if p and p.get("confidence", 0) >= self.threshold]
        return sorted(valid, key=lambda x: x.get("confidence", 0), reverse=True)[:Config.MAX_PREDICTIONS]
    
    def _build_prompt(self, event: Dict) -> str:
        home = event.get("home_team", "Local")
        away = event.get("away_team", "Visitante")
        league = event.get("league", "Desconocida")
        sport = event.get("sport", "Deporte")
        odds = event.get("odds", {})
        return f"""
Analiza este evento deportivo y genera predicciones detalladas en JSON.

DEPORTE: {sport}
LIGA: {league}
PARTIDO: {home} vs {away}

CUOTAS:
- Local: {odds.get('home_win', 'N/A')}
- Empate: {odds.get('draw', 'N/A')}
- Visitante: {odds.get('away_win', 'N/A')}

INSTRUCCIONES:
1. Usa tu conocimiento para estimar todas las métricas.
2. Para fútbol: goles, corners, tarjetas, faltas.
3. Da un análisis narrativo corto.
4. Confianza global (0-1) según calidad de datos.

RESPUESTE JSON EXACTO:
{{
    "event_id": "{event.get('id')}",
    "home_team": "{home}",
    "away_team": "{away}",
    "sport": "{sport}",
    "league": "{league}",
    "status": "{event.get('status', 'upcoming')}",
    "predictions": {{
        "winner_probability": {{"home_win": 0.0, "draw": 0.0, "away_win": 0.0}},
        "goals": {{"total_goals": 0.0, "home_goals": 0.0, "away_goals": 0.0, "over_under_line": 2.5, "over_probability": 0.0}},
        "corners": {{"total_corners": 0, "home_corners": 0, "away_corners": 0}},
        "cards": {{"total_cards": 0, "yellow_cards": 0, "red_cards": 0}},
        "fouls": {{"total_fouls": 0}},
        "recommended_bet": {{"type": "winner", "selection": "home", "value": ""}},
        "confidence_breakdown": {{"data_quality": 0.0, "market_consistency": 0.0, "historical_accuracy": 0.0, "overall_confidence": 0.0}}
    }},
    "analysis_summary": {{
        "key_factors": ["factor1", "factor2"],
        "match_narrative": "Breve análisis en español",
        "value_opportunity": "Mejor apuesta según cuotas"
    }}
}}"""

    def _parse(self, text: str, original: Dict) -> Optional[Dict]:
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            data = json.loads(cleaned)
            if "predictions" not in data:
                raise ValueError
            conf = data.get("predictions", {}).get("confidence_breakdown", {}).get("overall_confidence", 0.65)
            data["confidence"] = conf
            data["timestamp"] = datetime.now().isoformat()
            return data
        except:
            return None

# ==================== BOT DE TELEGRAM ====================
class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.app = None
    
    async def init(self):
        self.app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        self.app.add_handler(CommandHandler("start", self._start))
        self.app.add_handler(CommandHandler("status", self._status))
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        print("✅ Telegram bot listo")
    
    async def _start(self, update, context):
        await update.message.reply_text(
            "🤖 *Bot de Predicciones IA*\n\nAnalizo partidos con NVIDIA AI y te envío las mejores predicciones.\nUsa /status para ver configuración.",
            parse_mode="Markdown"
        )
    
    async def _status(self, update, context):
        msg = f"""
⚙️ *Estado*
- Modelo: `{Config.AI_MODEL}`
- Confianza mínima: {Config.CONFIDENCE_THRESHOLD*100}%
- Máx predicciones: {Config.MAX_PREDICTIONS}
- Deportes: {', '.join(Config.SPORTS_LEAGUES)}
        """
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def send_predictions(self, predictions: List[Dict]):
        if not predictions:
            await self._send("⚠️ No hay predicciones con suficiente confianza ahora.")
            return
        await self._send("🎯 *PREDICCIONES DEPORTIVAS IA*\n━━━━━━━━━━━━━━━━━━━━━━━")
        for i, p in enumerate(predictions, 1):
            msg = self._format(p, i)
            await self._send(msg)
    
    async def send_verification(self, event_id, predicted, actual):
        correct = self._check(predicted, actual)
        emoji = "✅" if correct else "❌"
        txt = f"📋 *VERIFICACIÓN* {emoji}\nEvento: `{event_id}`\nResultado: {'ACERTADA' if correct else 'FALLIDA'}"
        await self._send(txt)
    
    def _format(self, p: Dict, idx: int) -> str:
        preds = p.get("predictions", {})
        home = p.get("home_team", "Local")
        away = p.get("away_team", "Visitante")
        sport = p.get("sport", "Deporte")
        league = p.get("league", "")
        conf = p.get("confidence", 0)
        conf_emoji = "🟢" if conf>=0.8 else "🟡" if conf>=0.7 else "🟠" if conf>=0.6 else "🔴"
        
        wp = preds.get("winner_probability", {})
        goals = preds.get("goals", {})
        corners = preds.get("corners", {})
        cards = preds.get("cards", {})
        fouls = preds.get("fouls", {})
        narrative = p.get("analysis_summary", {}).get("match_narrative", "")
        value = p.get("analysis_summary", {}).get("value_opportunity", "")
        
        sport_emoji = {"football":"⚽","tennis":"🎾","basketball":"🏀","baseball":"⚾"}.get(sport.lower(), "🏆")
        
        msg = f"""
🎲 *Predicción #{idx}* | {sport_emoji} {sport.upper()}
━━━━━━━━━━━━━━━━━━━━━━━
*Partido:* {home} vs {away}
*Liga:* {league}
*Confianza:* {conf_emoji} {conf*100:.1f}%

📊 *PROBABILIDADES*
• 🏠 {home}: {wp.get('home_win',0)*100:.1f}%
• ⚖️ Empate: {wp.get('draw',0)*100:.1f}%
• ✈️ {away}: {wp.get('away_win',0)*100:.1f}%

⚽ *GOLES*
Total: {goals.get('total_goals',0):.1f} | {home}: {goals.get('home_goals',0):.1f} | {away}: {goals.get('away_goals',0):.1f}
Over {goals.get('over_under_line',2.5)}: {goals.get('over_probability',0)*100:.0f}%

🚩 *CÓRNERS*: {corners.get('total_corners',0)} ({home}:{corners.get('home_corners',0)} / {away}:{corners.get('away_corners',0)})
🟨 *TARJETAS*: {cards.get('total_cards',0)} (A:{cards.get('yellow_cards',0)} R:{cards.get('red_cards',0)})
💥 *FALTAS*: {fouls.get('total_fouls',0)}

💡 *ANÁLISIS*
{narrative[:300]}

🎯 *Valor:* {value if value else "No especificado"}
━━━━━━━━━━━━━━━━━━━━━━━
"""
        return msg
    
    async def _send(self, text):
        max_len = 4096
        if len(text) <= max_len:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
        else:
            for i in range(0, len(text), max_len):
                await self.bot.send_message(chat_id=self.chat_id, text=text[i:i+max_len], parse_mode="Markdown")
                await asyncio.sleep(0.5)
    
    def _check(self, predicted, actual):
        bet = predicted.get("predictions", {}).get("recommended_bet", {})
        bet_type = bet.get("type", "")
        bet_sel = bet.get("selection", "")
        actual_res = actual.get("result", "").lower()
        if bet_type == "winner":
            if bet_sel == "home" and "home" in actual_res: return True
            if bet_sel == "away" and "away" in actual_res: return True
            if bet_sel == "draw" and "draw" in actual_res: return True
        return False
    
    async def shutdown(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

# ==================== VERIFICADOR DE RESULTADOS ====================
class Verifier:
    def __init__(self, api_client: OddsClient, telegram_bot: TelegramBot):
        self.api = api_client
        self.bot = telegram_bot
        self.pending = {}
    
    def register(self, event_id: str, prediction: Dict):
        if event_id not in self.pending:
            self.pending[event_id] = {"prediction": prediction, "status": "pending"}
            print(f"📝 Registrada predicción para {event_id}")
    
    async def check_pending(self):
        to_remove = []
        for eid, data in self.pending.items():
            if data["status"] != "pending":
                to_remove.append(eid)
                continue
            result = await self.api.verify_result(eid)
            if result and result.get("status") == "finished":
                await self.bot.send_verification(eid, data["prediction"], result)
                data["status"] = "verified"
                to_remove.append(eid)
        for eid in to_remove:
            del self.pending[eid]
    
    async def run_loop(self, interval=3600):
        while True:
            await self.check_pending()
            await asyncio.sleep(interval)

# ==================== BOT PRINCIPAL ====================
class PredictionBot:
    def __init__(self):
        self.running = True
        self.api_client = OddsClient()
        self.ai = AIAnalyzer()
        self.telegram = TelegramBot()
        self.verifier = Verifier(self.api_client, self.telegram)
    
    async def start(self):
        Config.validate()
        print("🚀 Iniciando bot en Railway...")
        await self.telegram.init()
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        await asyncio.gather(self._analysis_loop(), self.verifier.run_loop())
    
    async def _analysis_loop(self):
        while self.running:
            try:
                print(f"\n📊 [{datetime.now()}] Analizando eventos...")
                live = await self.api_client.get_live_events()
                upcoming = await self.api_client.get_upcoming_events()
                all_events = live + upcoming
                print(f"📡 Eventos: {len(all_events)} (vivos:{len(live)}, próximos:{len(upcoming)})")
                
                if not all_events:
                    await asyncio.sleep(60)
                    continue
                
                predictions = []
                for ev in all_events[:20]:
                    print(f"🤖 Analizando {ev.get('home_team')} vs {ev.get('away_team')}")
                    pred = await self.ai.analyze(ev)
                    if pred:
                        predictions.append(pred)
                    await asyncio.sleep(2)
                
                ranked = await self.ai.rank(predictions)
                print(f"🏆 Mejores: {len(ranked)}")
                for p in ranked:
                    self.verifier.register(p.get("event_id"), p)
                await self.telegram.send_predictions(ranked)
                
                wait = Config.ANALYSIS_INTERVAL_SECONDS
                if live:
                    wait = min(wait, 1800)
                print(f"⏳ Esperando {wait//60} minutos...")
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"❌ Error en análisis: {e}")
                await asyncio.sleep(60)
    
    def _stop(self, *args):
        print("🛑 Deteniendo bot...")
        self.running = False
        asyncio.create_task(self.telegram.shutdown())
        sys.exit(0)

async def main():
    bot = PredictionBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
