#!/usr/bin/env python3
"""
Bot de predicciones deportivas con IA (Perplexity Sonar Pro Search)
- Análisis por deporte
- Mensajes cortos y precisos
- Saldo simulado
- Verificación automática de resultados
"""

import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
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

    DEFAULT_LEAGUES = ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB", "MLS"]
    SPORTS_LEAGUES = os.environ.get("SPORTS_LEAGUES", ",".join(DEFAULT_LEAGUES)).split(",")

    AI_MODEL = "perplexity/sonar-pro-search"
    MAX_PREDICTIONS = int(os.environ.get("MAX_PREDICTIONS", "5"))   # menos predicciones para no saturar
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
    ANALYSIS_INTERVAL_SECONDS = int(os.environ.get("ANALYSIS_INTERVAL_SECONDS", "7200"))  # 2 horas
    MAX_EVENTS_PER_CYCLE = int(os.environ.get("MAX_EVENTS_PER_CYCLE", "15"))

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "https://railway.app")
    OPENROUTER_SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "Sports Prediction Bot")

    # Mapeo leagueID -> deporte (para saber qué métricas mostrar)
    LEAGUE_TO_SPORT = {
        "NFL": "football", "NBA": "basketball", "MLB": "baseball", "NHL": "hockey",
        "NCAAF": "football", "NCAAB": "basketball", "MLS": "football"
    }

    # Saldo simulado inicial
    INITIAL_BALANCE = 1000.0
    BET_AMOUNT = 100.0   # apuesta fija por cada predicción

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


# ==================== CLIENTE API (igual que antes, con rate limit) ====================
class OddsAPIClient:
    def __init__(self):
        self.base_url = "https://api.sportsgameodds.com/v2"
        self.api_key = Config.SPORTS_ODDS_API_KEY
        self.league_ids = Config.SPORTS_LEAGUES
        self.headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        self.request_delay = 1.5

    async def _request_with_retry(self, url: str, params: dict, max_retries: int = 3) -> Optional[Dict]:
        for attempt in range(max_retries):
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait = 2 ** attempt
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

    async def get_finished_events(self) -> List[Dict[str, Any]]:
        """Para verificación: eventos finalizados recientemente (últimas 24h)"""
        await asyncio.sleep(self.request_delay)
        events = await self.fetch_all_events("events?status=finished")
        return [self._normalize_event(e) for e in events]

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

        # Extraer cuotas simplificadas (puedes mejorarlo)
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
            "score": event.get("score", {})  # para resultados reales
        }


# ==================== ANALIZADOR IA (Perplexity Sonar Pro Search) ====================
class AIAnalyzer:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=Config.OPENROUTER_BASE_URL,
            api_key=Config.OPENROUTER_API_KEY,
            timeout=60
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
                    {"role": "system", "content": "Eres un analista deportivo. Buscas información actualizada y respondes ÚNICAMENTE con un objeto JSON válido, sin texto adicional."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3000,   # reducido para respuestas más cortas
                extra_headers=self.headers
            )
            if not response or not response.choices:
                print("❌ Respuesta vacía de la IA")
                return None
            raw = response.choices[0].message.content
            return self._parse(raw, event)
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

        # Instrucciones según el deporte para que el modelo solo devuelva métricas relevantes
        if sport == "football":
            metrics = "goles totales, goles de cada equipo, número de corners, tarjetas amarillas/rojas, faltas"
        elif sport == "basketball":
            metrics = "puntos totales, puntos de cada equipo, rebotes totales, asistencias totales"
        elif sport == "baseball":
            metrics = "carreras totales, carreras de cada equipo, hits, errores"
        elif sport == "hockey":
            metrics = "goles totales, goles de cada equipo, tiros a puerta, minutos de penalización"
        else:
            metrics = "métricas estándar del deporte"

        return f"""
Activa la búsqueda en internet. Analiza el siguiente evento de {sport} (liga {league}):

{home} vs {away}

Cuotas aproximadas: Local {odds.get('home_win','N/A')} | Empate {odds.get('draw','N/A')} | Visitante {odds.get('away_win','N/A')}

Primero busca: lesiones, forma reciente (últimos 5 partidos), historial directo, noticias.

Luego genera un JSON con:
- Probabilidades de victoria (home, draw, away)
- Las métricas específicas para {sport}: {metrics}
- Una recomendación de apuesta (tipo y selección)
- Un nivel de confianza global (0-1) basado en la calidad de la información encontrada

No inventes métricas que no correspondan al deporte. Si no encuentras datos para alguna métrica, pon 0 o null.

Responde ÚNICAMENTE con este JSON (sin texto extra):
{{
    "event_id": "{event.get('id')}",
    "home_team": "{home}",
    "away_team": "{away}",
    "sport": "{sport}",
    "league": "{league}",
    "start_time": "{event.get('start_time')}",
    "predictions": {{
        "winner_probability": {{"home_win": 0.0, "draw": 0.0, "away_win": 0.0}},
        "sport_metrics": {{
            "total_goals": 0.0, "home_goals": 0.0, "away_goals": 0.0,
            "corners": 0, "yellow_cards": 0, "red_cards": 0, "fouls": 0,
            "total_points": 0, "home_points": 0, "away_points": 0,
            "rebounds": 0, "assists": 0,
            "runs": 0, "home_runs": 0, "away_runs": 0, "hits": 0, "errors": 0,
            "shots": 0, "penalty_minutes": 0
        }},
        "recommended_bet": {{"type": "winner", "selection": "home", "value": ""}},
        "confidence_breakdown": {{
            "data_quality": 0.0,
            "market_consistency": 0.0,
            "historical_accuracy": 0.0,
            "overall_confidence": 0.0
        }}
    }},
    "analysis_summary": {{
        "key_factors": ["factor1", "factor2"],
        "value_opportunity": "explicación breve"
    }}
}}
"""

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
            # Extraer confianza global
            conf = data.get("predictions", {}).get("confidence_breakdown", {}).get("overall_confidence", 0.65)
            data["confidence"] = conf
            # Asegurar que exista sport_metrics
            if "sport_metrics" not in data["predictions"]:
                data["predictions"]["sport_metrics"] = {}
            return data
        except Exception as e:
            print(f"⚠️ Error parseando JSON: {e}")
            return None


# ==================== GESTIÓN DE SALDO SIMULADO Y VERIFICACIÓN ====================
class BettingSimulator:
    def __init__(self):
        self.balance = Config.INITIAL_BALANCE
        self.active_bets = {}  # event_id -> {prediction, bet_amount, selected_winner, odds}

    def place_bet(self, event_id: str, prediction: Dict, home_odds: float, away_odds: float, draw_odds: float) -> bool:
        """Registra una apuesta simulada (si hay saldo suficiente)"""
        if self.balance < Config.BET_AMOUNT:
            return False
        # Determinar la selección recomendada
        rec = prediction.get("predictions", {}).get("recommended_bet", {})
        bet_type = rec.get("type", "winner")
        selection = rec.get("selection", "home")
        # Asignar cuota correspondiente
        if selection == "home":
            odds = home_odds if home_odds != "N/A" else 2.0
        elif selection == "away":
            odds = away_odds if away_odds != "N/A" else 2.0
        elif selection == "draw":
            odds = draw_odds if draw_odds != "N/A" else 3.0
        else:
            odds = 2.0

        self.active_bets[event_id] = {
            "prediction": prediction,
            "amount": Config.BET_AMOUNT,
            "selection": selection,
            "odds": float(odds) if isinstance(odds, (int, float)) else 2.0,
            "bet_type": bet_type
        }
        self.balance -= Config.BET_AMOUNT
        return True

    def settle_bet(self, event_id: str, actual_winner: str) -> float:
        """Liquida una apuesta. actual_winner = 'home', 'away', 'draw' """
        if event_id not in self.active_bets:
            return 0.0
        bet = self.active_bets.pop(event_id)
        if bet["selection"] == actual_winner:
            winnings = bet["amount"] * bet["odds"]
            self.balance += winnings
            return winnings
        else:
            return 0.0  # ya se descontó al apostar

    def get_balance(self) -> float:
        return self.balance


# ==================== TELEGRAM BOT (mensajes cortos) ====================
class TelegramBot:
    def __init__(self, simulator: BettingSimulator):
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.app = None
        self.simulator = simulator

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
        self.app.add_handler(CommandHandler("balance", self._balance))
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True, allowed_updates=None)
        print("✅ Telegram bot ready")

    async def _start(self, update, context):
        await update.message.reply_text(
            "🤖 *Bot de Predicciones IA* (Perplexity Sonar Pro Search)\n"
            "Analizo partidos en vivo/próximos y envío las mejores predicciones.\n"
            "Comandos:\n/status - Configuración\n/balance - Saldo simulado",
            parse_mode="Markdown"
        )

    async def _status(self, update, context):
        msg = f"""
⚙️ *Estado*
- Modelo: `{Config.AI_MODEL}`
- Confianza mínima: {Config.CONFIDENCE_THRESHOLD*100}%
- Máx predicciones: {Config.MAX_PREDICTIONS}
- Ligas: {', '.join(Config.SPORTS_LEAGUES)}
- Saldo actual: ${self.simulator.get_balance():.2f}
        """
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _balance(self, update, context):
        await update.message.reply_text(f"💰 *Saldo simulado*: ${self.simulator.get_balance():.2f}", parse_mode="Markdown")

    async def send_prediction(self, prediction: Dict, event: Dict):
        """Envía un mensaje corto con los datos esenciales"""
        home = event.get("home_team")
        away = event.get("away_team")
        sport = prediction.get("sport", "deporte")
        league = prediction.get("league", "")
        start_time_str = event.get("start_time")
        # Convertir UTC a hora Argentina (UTC-3)
        if start_time_str:
            try:
                dt_utc = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                dt_arg = dt_utc.astimezone(timezone(timedelta(hours=-3)))
                hora_local = dt_arg.strftime("%d/%m %H:%M")
            except:
                hora_local = "Fecha no disponible"
        else:
            hora_local = "Sin hora"

        pred_data = prediction.get("predictions", {})
        winner_probs = pred_data.get("winner_probability", {})
        home_prob = winner_probs.get("home_win", 0) * 100
        draw_prob = winner_probs.get("draw", 0) * 100
        away_prob = winner_probs.get("away_win", 0) * 100

        # Determinar favorito
        max_prob = max(home_prob, draw_prob, away_prob)
        if max_prob == home_prob:
            favorite = f"🏠 {home}"
            winner_code = "home"
        elif max_prob == away_prob:
            favorite = f"✈️ {away}"
            winner_code = "away"
        else:
            favorite = "⚖️ Empate"
            winner_code = "draw"

        # Métricas específicas según deporte
        metrics = pred_data.get("sport_metrics", {})
        extra_line = ""
        if sport == "football":
            goles = metrics.get("total_goals", 0)
            corners = metrics.get("corners", 0)
            cards = metrics.get("yellow_cards", 0) + metrics.get("red_cards", 0)
            extra_line = f"⚽ {goles:.1f} g | 🚩 {corners} có | 🟨 {cards} tar"
        elif sport == "basketball":
            pts = metrics.get("total_points", 0)
            reb = metrics.get("rebounds", 0)
            ast = metrics.get("assists", 0)
            extra_line = f"🏀 {pts:.0f} pts | {reb} reb | {ast} ast"
        elif sport == "baseball":
            runs = metrics.get("runs", 0)
            hits = metrics.get("hits", 0)
            extra_line = f"⚾ {runs} carr | {hits} hits"
        elif sport == "hockey":
            goals = metrics.get("total_goals", 0)
            shots = metrics.get("shots", 0)
            pims = metrics.get("penalty_minutes", 0)
            extra_line = f"🏒 {goals} g | {shots} shots | {pims}' pen"

        confianza = prediction.get("confidence", 0) * 100
        narrative = prediction.get("analysis_summary", {}).get("value_opportunity", "")
        if len(narrative) > 60:
            narrative = narrative[:57] + "..."

        # Colocar apuesta simulada (extraer cuotas del evento)
        home_odds = event.get("odds", {}).get("home_win", "2.0")
        away_odds = event.get("odds", {}).get("away_win", "2.0")
        draw_odds = event.get("odds", {}).get("draw", "3.0")
        try:
            home_odds_f = float(home_odds) if home_odds != "N/A" else 2.0
            away_odds_f = float(away_odds) if away_odds != "N/A" else 2.0
            draw_odds_f = float(draw_odds) if draw_odds != "N/A" else 3.0
        except:
            home_odds_f = away_odds_f = 2.0
            draw_odds_f = 3.0

        bet_placed = self.simulator.place_bet(event.get("id"), prediction, home_odds_f, away_odds_f, draw_odds_f)
        if not bet_placed:
            saldo_msg = "⚠️ Saldo insuficiente para apostar"
        else:
            saldo_msg = f"💰 Apuesta ${Config.BET_AMOUNT:.0f} a {favorite} | Saldo: ${self.simulator.get_balance():.2f}"

        msg = f"""
🎲 *{home} vs {away}* | {league} | {hora_local}
📊 *Favorito:* {favorite} ({max_prob:.0f}%)
{extra_line}
🎯 *Confianza:* {confianza:.0f}%
💡 *Valor:* {narrative}
{saldo_msg}
        """
        # Dividir si es muy largo (no debería)
        if len(msg) > 4000:
            msg = msg[:4000]
        await self.bot.send_message(chat_id=self.chat_id, text=msg, parse_mode="Markdown")

    async def send_result(self, event_id: str, actual_winner: str, prediction: Dict, winnings: float):
        """Notifica el resultado de una apuesta"""
        home = prediction.get("home_team")
        away = prediction.get("away_team")
        if winnings > 0:
            emoji = "✅"
            text = f"¡ACERTASTE! Ganaste ${winnings:.2f}"
        else:
            emoji = "❌"
            text = f"Fallaste. Perdiste ${Config.BET_AMOUNT:.2f}"
        saldo = self.simulator.get_balance()
        msg = f"{emoji} *{home} vs {away}*\n{text}\n💰 Nuevo saldo: ${saldo:.2f}"
        await self.bot.send_message(chat_id=self.chat_id, text=msg, parse_mode="Markdown")

    async def shutdown(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()


# ==================== VERIFICADOR DE RESULTADOS ====================
class ResultVerifier:
    def __init__(self, api_client: OddsAPIClient, simulator: BettingSimulator, telegram_bot: TelegramBot):
        self.api = api_client
        self.simulator = simulator
        self.telegram = telegram_bot

    async def check_pending_bets(self):
        """Busca eventos finalizados y liquida apuestas"""
        finished = await self.api.get_finished_events()
        for event in finished:
            event_id = event.get("id")
            if event_id in self.simulator.active_bets:
                # Determinar el ganador real desde el score (simplificado)
                # En una API real vendría un campo 'winner'. Aquí asumimos que score tiene "home", "away"
                score = event.get("score", {})
                home_score = score.get("home", 0)
                away_score = score.get("away", 0)
                if home_score > away_score:
                    actual = "home"
                elif away_score > home_score:
                    actual = "away"
                else:
                    actual = "draw"
                bet_info = self.simulator.active_bets[event_id]
                winnings = self.simulator.settle_bet(event_id, actual)
                await self.telegram.send_result(event_id, actual, bet_info["prediction"], winnings)

    async def run_loop(self, interval=1800):  # cada 30 minutos
        while True:
            await self.check_pending_bets()
            await asyncio.sleep(interval)


# ==================== BOT PRINCIPAL ====================
class PredictionBot:
    def __init__(self):
        self.running = True
        self.api = OddsAPIClient()
        self.ai = AIAnalyzer()
        self.simulator = BettingSimulator()
        self.telegram = TelegramBot(self.simulator)
        self.verifier = ResultVerifier(self.api, self.simulator, self.telegram)

    async def start(self):
        Config.validate()
        print("🚀 Bot iniciando en Railway con Perplexity Sonar Pro Search...")
        await self.telegram.init()
        await asyncio.sleep(10)
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)
        # Ejecutar análisis y verificación en paralelo
        await asyncio.gather(
            self._analysis_loop(),
            self.verifier.run_loop()
        )

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
                for ev in all_events[:Config.MAX_EVENTS_PER_CYCLE]:
                    print(f"🤖 Analizando {ev.get('home_team')} vs {ev.get('away_team')}")
                    pred = await self.ai.analyze(ev)
                    if pred:
                        predictions.append(pred)
                    await asyncio.sleep(2)

                ranked = await self.ai.rank(predictions)
                for pred in ranked:
                    # Buscar el evento original correspondiente (para tener cuotas y hora)
                    event_orig = next((e for e in all_events if e.get("id") == pred.get("event_id")), None)
                    if event_orig:
                        await self.telegram.send_prediction(pred, event_orig)

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
