#!/usr/bin/env python3
import asyncio
import json
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

import aiohttp
from openai import AsyncOpenAI
from telegram import Bot
from telegram.ext import Application, CommandHandler

# ==================== CONFIGURACIÓN ====================
class Config:
    SPORTS_ODDS_API_KEY = os.environ.get("SPORTS_ODDS_API_KEY_HEADER")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    # IDs correctos para plan gratuito (UCL no funciona, lo quitamos)
    DEFAULT_LEAGUES = ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB", "MLS"]
    SPORTS_LEAGUES = os.environ.get("SPORTS_LEAGUES", ",".join(DEFAULT_LEAGUES)).split(",")

    AI_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    MAX_PREDICTIONS = int(os.environ.get("MAX_PREDICTIONS", "10"))
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.65"))
    ANALYSIS_INTERVAL_SECONDS = int(os.environ.get("ANALYSIS_INTERVAL_SECONDS", "3600"))

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://railway.app")
    OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "Sports Prediction Bot")

    LEAGUE_TO_SPORT = {
        "NFL": "football", "NBA": "basketball", "MLB": "baseball", "NHL": "hockey",
        "NCAAF": "football", "NCAAB": "basketball", "MLS": "football"
    }

    @classmethod
    def validate(cls):
        missing = []
        if not cls.SPORTS_ODDS_API_KEY: missing.append("SPORTS_ODDS_API_KEY_HEADER")
        if not cls.OPENROUTER_API_KEY: missing.append("OPENROUTER_API_KEY")
        if not cls.TELEGRAM_BOT_TOKEN: missing.append("TELEGRAM_BOT_TOKEN")
        if not cls.TELEGRAM_CHAT_ID: missing.append("TELEGRAM_CHAT_ID")
        if missing:
            raise ValueError(f"Faltan variables: {', '.join(missing)}")
        return True

# ==================== CLIENTE API CON RATE LIMIT ====================
class OddsAPIClient:
    def __init__(self):
        self.base_url = "https://api.sportsgameodds.com/v2"
        self.api_key = Config.SPORTS_ODDS_API_KEY
        self.league_ids = Config.SPORTS_LEAGUES
        self.headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        self.request_delay = 1.5  # segundos entre peticiones

    async def _request_with_retry(self, url: str, params: dict, max_retries: int = 3) -> Optional[Dict]:
        for attempt in range(max_retries):
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait = 2 ** attempt  # 1, 2, 4 segundos
                        print(f"⚠️ Rate limit. Reintentando en {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        error_text = await resp.text()
                        print(f"❌ API error {resp.status}: {error_text}")
                        return None
        return None

    async def fetch_all_events(self, endpoint: str) -> List[Dict]:
        all_events = []
        next_cursor = None
        leagues_param = ",".join(self.league_ids)

        while True:
            params = {"apiKey": self.api_key, "leagueID": leagues_param, "oddsAvailable": "true"}
            if next_cursor:
                params["nextCursor"] = next_cursor

            url = f"{self.base_url}/{endpoint}"
            data = await self._request_with_retry(url, params)
            if not data or not data.get("success"):
                break

            page_events = data.get("data", [])
            all_events.extend(page_events)
            next_cursor = data.get("nextCursor")
            if not next_cursor:
                break
            await asyncio.sleep(self.request_delay)

        return [e for e in all_events if e.get("leagueID") in self.league_ids]

    async def get_live_events(self) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.5)
        events = await self.fetch_all_events("events?status=live")
        return [self._normalize_event(e) for e in events]

    async def get_upcoming_events(self) -> List[Dict[str, Any]]:
        await asyncio.sleep(self.request_delay)
        events = await self.fetch_all_events("events?status=upcoming")
        return [self._normalize_event(e) for e in events]

    async def get_event_details(self, event_id: str) -> Optional[Dict]:
        return None

    async def verify_result(self, event_id: str) -> Optional[Dict]:
        return None

    def _normalize_event(self, event: Dict) -> Dict:
        teams = event.get("teams", {})
        home_data = teams.get("home", {})
        away_data = teams.get("away", {})

        home_name = (home_data.get("names", {}).get("long") or
                     home_data.get("names", {}).get("medium") or
                     home_data.get("names", {}).get("short") or "Unknown")
        away_name = (away_data.get("names", {}).get("long") or
                     away_data.get("names", {}).get("medium") or
                     away_data.get("names", {}).get("short") or "Unknown")

        league_id = event.get("leagueID")
        sport = Config.LEAGUE_TO_SPORT.get(league_id, "deporte")
        status_info = event.get("status", {})
        odds_info = event.get("odds", {})

        home_odds = away_odds = draw_odds = "N/A"
        for odd_id, odd_val in odds_info.items():
            if "moneyline" in odd_id.lower() or "h2h" in odd_id.lower():
                home_odds = odd_val.get("bookOdds", "N/A")
                if "homeOdds" in odd_val:
                    home_odds = odd_val["homeOdds"]
                if "awayOdds" in odd_val:
                    away_odds = odd_val["awayOdds"]
                break

        return {
            "id": event.get("eventID"),
            "sport": sport,
            "league": league_id,
            "home_team": home_name,
            "away_team": away_name,
            "start_time": status_info.get("startsAt"),
            "status": status_info.get("displayLong", ""),
            "odds": {"home_win": home_odds, "draw": draw_odds, "away_win": away_odds},
            "score": {}
        }

# ==================== ANALIZADOR IA ====================
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
                    {"role": "system", "content": "Eres analista deportivo. Responde SOLO con JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=4000,
                extra_headers=self.headers
            )
            return self._parse(response.choices[0].message.content, event)
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
        sport = event.get("sport", "deporte")
        odds = event.get("odds", {})
        return f"""
Analiza este evento deportivo y genera predicciones detalladas en JSON.

DEPORTE: {sport}
LIGA: {league}
PARTIDO: {home} vs {away}
CUOTAS: Local {odds.get('home_win','N/A')} Empate {odds.get('draw','N/A')} Visitante {odds.get('away_win','N/A')}

RESPUESTA JSON:
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
        "match_narrative": "breve análisis en español",
        "value_opportunity": "mejor apuesta"
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
            conf = data.get("predictions", {}).get("confidence_breakdown", {}).get("overall_confidence", 0.65)
            data["confidence"] = conf
            return data
        except:
            return None

# ==================== TELEGRAM BOT ====================
class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.app = None

    async def init(self):
        self.app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        for attempt in range(3):
            try:
                await self.app.bot.delete_webhook(drop_pending_updates=True)
                print("Webhook eliminado")
                break
            except Exception as e:
                print(f"Intento {attempt+1} falló: {e}")
                await asyncio.sleep(2)
        await asyncio.sleep(2)
        try:
            await self.app.bot.get_updates(offset=-1, timeout=1)
        except Exception:
            pass
        self.app.add_handler(CommandHandler("start", self._start))
        self.app.add_handler(CommandHandler("status", self._status))
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True, allowed_updates=None)
        print("✅ Telegram bot ready")

    async def _start(self, update, context):
        await update.message.reply_text("🤖 Bot de predicciones IA activo.\nUsa /status para ver configuración.", parse_mode="Markdown")

    async def _status(self, update, context):
        msg = f"""
⚙️ *Estado*
- Modelo: `{Config.AI_MODEL}`
- Confianza mínima: {Config.CONFIDENCE_THRESHOLD*100}%
- Máx predicciones: {Config.MAX_PREDICTIONS}
- Ligas: {', '.join(Config.SPORTS_LEAGUES)}
        """
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def send_predictions(self, predictions: List[Dict]):
        if not predictions:
            await self._send("⚠️ Sin predicciones con suficiente confianza ahora.")
            return
        await self._send("🎯 *PREDICCIONES DEPORTIVAS IA*\n━━━━━━━━━━━━━━━━━━━━━━━")
        for i, p in enumerate(predictions[:Config.MAX_PREDICTIONS], 1):
            msg = self._format(p, i)
            await self._send(msg)

    def _format(self, p: Dict, idx: int) -> str:
        preds = p.get("predictions", {})
        home = p.get("home_team", "Local")
        away = p.get("away_team", "Visitante")
        sport = p.get("sport", "deporte")
        league = p.get("league", "")
        conf = p.get("confidence", 0)
        conf_emoji = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.7 else "🟠" if conf >= 0.6 else "🔴"
        wp = preds.get("winner_probability", {})
        goals = preds.get("goals", {})
        corners = preds.get("corners", {})
        cards = preds.get("cards", {})
        fouls = preds.get("fouls", {})
        narrative = p.get("analysis_summary", {}).get("match_narrative", "")
        value = p.get("analysis_summary", {}).get("value_opportunity", "")
        sport_emoji = {"football": "⚽", "basketball": "🏀", "baseball": "⚾", "hockey": "🏒", "deporte": "🏆"}.get(sport.lower(), "🏆")
        msg = f"""
🎲 *Predicción #{idx}* | {sport_emoji} {sport.upper()}
━━━━━━━━━━━━━━━━━━━━━━━
*Partido:* {home} vs {away}
*Liga:* {league}
*Confianza:* {conf_emoji} {conf*100:.1f}%

📊 *PROBABILIDADES*
• 🏠 {home}: {wp.get('home_win', 0)*100:.1f}%
• ⚖️ Empate: {wp.get('draw', 0)*100:.1f}%
• ✈️ {away}: {wp.get('away_win', 0)*100:.1f}%

⚽ *GOLES*
Total: {goals.get('total_goals', 0):.1f} | {home}: {goals.get('home_goals', 0):.1f} | {away}: {goals.get('away_goals', 0):.1f}
Over {goals.get('over_under_line', 2.5)}: {goals.get('over_probability', 0)*100:.0f}%

🚩 *CÓRNERS*: {corners.get('total_corners', 0)} ({home}:{corners.get('home_corners', 0)} / {away}:{corners.get('away_corners', 0)})
🟨 *TARJETAS*: {cards.get('total_cards', 0)} (A:{cards.get('yellow_cards', 0)} R:{cards.get('red_cards', 0)})
💥 *FALTAS*: {fouls.get('total_fouls', 0)}

💡 *ANÁLISIS*
{narrative[:300]}

🎯 *Valor:* {value if value else "No especificado"}
━━━━━━━━━━━━━━━━━━━━━━━
"""
        return msg

    async def _send(self, text: str):
        max_len = 4096
        if len(text) <= max_len:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
        else:
            for i in range(0, len(text), max_len):
                await self.bot.send_message(chat_id=self.chat_id, text=text[i:i+max_len], parse_mode="Markdown")
                await asyncio.sleep(0.5)

    async def shutdown(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

# ==================== VERIFICADOR ====================
class Verifier:
    def __init__(self, api_client: OddsAPIClient, telegram_bot: TelegramBot):
        self.api = api_client
        self.bot = telegram_bot
        self.pending = {}

    def register(self, event_id: str, prediction: Dict):
        if event_id not in self.pending:
            self.pending[event_id] = prediction

    async def run_loop(self, interval: int = 3600):
        while True:
            await asyncio.sleep(interval)

# ==================== BOT PRINCIPAL ====================
class PredictionBot:
    def __init__(self):
        self.running = True
        self.api = OddsAPIClient()
        self.ai = AIAnalyzer()
        self.telegram = TelegramBot()
        self.verifier = Verifier(self.api, self.telegram)

    async def start(self):
        Config.validate()
        print("🚀 Bot iniciando en Railway...")
        await self.telegram.init()
        await asyncio.sleep(10)  # Estabilizar Telegram
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        await self._analysis_loop()

    async def _analysis_loop(self):
        while self.running:
            try:
                print(f"\n📊 [{datetime.now()}] Obteniendo eventos...")
                live = await self.api.get_live_events()
                upcoming = await self.api.get_upcoming_events()
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
                for p in ranked:
                    self.verifier.register(p.get("event_id"), p)

                await self.telegram.send_predictions(ranked)

                wait = Config.ANALYSIS_INTERVAL_SECONDS
                if live:
                    wait = min(wait, 1800)
                print(f"⏳ Esperando {wait//60} minutos...")
                await asyncio.sleep(wait)

            except Exception as e:
                print(f"❌ Error en ciclo: {e}")
                await asyncio.sleep(60)

    def _stop(self, *args):
        print("🛑 Apagando bot...")
        self.running = False
        asyncio.create_task(self.telegram.shutdown())
        sys.exit(0)

async def main():
    bot = PredictionBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
