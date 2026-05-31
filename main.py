import os
import io
import logging
import base64
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from openai import OpenAI

# ================= LOGGING =================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================= VARIABLES DE ENTORNO =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
# Opcional: si quieres usar 2Captcha como respaldo (descomentar luego)
# TWOCAPTCHA_API_KEY = os.getenv("TWOCAPTCHA_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("❌ Falta TELEGRAM_TOKEN")
if not SILICONFLOW_API_KEY:
    raise ValueError("❌ Falta SILICONFLOW_API_KEY")

# ================= CONFIGURACIÓN DE IA =================
client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)
MODELO_CAPTCHA = "Qwen/Qwen-Image"   # Modelo para leer texto en imágenes

# ================= ESTADO DEL BOT =================
bot_paused = False   # Si está en pausa, no procesa captchas

# ================= FUNCIONES DE AYUDA =================
def image_to_base64(image_bytes: bytes) -> str:
    """Convierte bytes de imagen a base64 para enviar a SiliconFlow."""
    return base64.b64encode(image_bytes).decode('utf-8')

async def resolver_captcha_con_ia(image_bytes: bytes) -> str | None:
    """
    Envía la imagen a SiliconFlow (Qwen/Qwen-Image) y devuelve el texto del captcha.
    """
    b64_image = image_to_base64(image_bytes)
    prompt = (
        "Eres un lector de captchas. La imagen contiene un captcha de LETRAS (sin números). "
        "Responde ÚNICAMENTE con el texto exacto que ves, sin explicaciones. "
        "Si no ves texto claro, responde 'ERROR'."
    )
    try:
        response = client.chat.completions.create(
            model=MODELO_CAPTCHA,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=20
        )
        texto = response.choices[0].message.content.strip()
        logger.info(f"IA respondió: {texto}")
        return texto if texto != "ERROR" else None
    except Exception as e:
        logger.error(f"Error llamando a SiliconFlow: {e}")
        return None

# ================= MANEJADORES DE TELAGRAM =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start"""
    await update.message.reply_text(
        "🤖 Bot Captcha Solver activo.\n\n"
        "Envíame una imagen de un captcha (solo letras) y te devolveré el texto.\n"
        "Luego podrás confirmar si es correcto con los botones.\n\n"
        "Comandos:\n"
        "/pause - Pausar el bot\n"
        "/resume - Reanudar el bot\n"
        "/status - Ver estado actual"
    )

async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pausa el bot (no procesa captchas)"""
    global bot_paused
    bot_paused = True
    await update.message.reply_text("⏸ Bot pausado. No procesaré nuevos captchas hasta /resume.")

async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reanuda el bot"""
    global bot_paused
    bot_paused = False
    await update.message.reply_text("▶️ Bot reanudado. Ya puedo procesar captchas.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra estado actual"""
    estado = "PAUSADO" if bot_paused else "ACTIVO"
    await update.message.reply_text(f"📊 Estado del bot: {estado}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Procesa la foto enviada (captcha)"""
    global bot_paused
    if bot_paused:
        await update.message.reply_text("⏸ El bot está pausado. Usa /resume para activarlo.")
        return

    # Obtener la foto de mayor resolución
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()

    # Notificar que se está procesando
    waiting_msg = await update.message.reply_text("🔍 Analizando captcha...")

    # Llamar a la IA
    texto = await resolver_captcha_con_ia(bytes(image_bytes))
    if not texto:
        await waiting_msg.edit_text("❌ No pude leer el captcha. Intenta con otra imagen más clara.")
        return

    # Construir teclado con opciones
    keyboard = [
        [
            InlineKeyboardButton("✅ Correcto", callback_data=f"correcto_{texto}"),
            InlineKeyboardButton("❌ Incorrecto", callback_data=f"incorrecto_{texto}")
        ],
        [InlineKeyboardButton("🔄 Reintentar con esta imagen", callback_data="reintentar")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Guardar la imagen en el contexto para posible reintento
    context.user_data['last_image_bytes'] = image_bytes
    context.user_data['last_texto'] = texto

    await waiting_msg.edit_text(
        f"📝 Solución propuesta:\n\n`{texto}`\n\n¿Es correcto?",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja los botones de confirmación/reintento"""
    query = update.callback_query
    await query.answer()

    data = query.data
    if data.startswith("correcto_"):
        texto = data.split("_", 1)[1]
        await query.edit_message_text(
            f"✅ ¡Perfecto! La solución `{texto}` ha sido marcada como correcta.\n"
            "Gracias por la validación.",
            parse_mode='Markdown'
        )
        # Aquí podrías guardar en una base de datos que la solución fue buena

    elif data.startswith("incorrecto_"):
        texto = data.split("_", 1)[1]
        await query.edit_message_text(
            f"❌ Marcaste como incorrecto: `{texto}`.\n"
            "Si quieres, puedes enviarme la misma imagen de nuevo o probar con otra.",
            parse_mode='Markdown'
        )

    elif data == "reintentar":
        # Recuperar la imagen guardada y volver a intentar
        image_bytes = context.user_data.get('last_image_bytes')
        if not image_bytes:
            await query.edit_message_text("⚠️ No hay imagen guardada para reintentar. Envía una nueva.")
            return
        await query.edit_message_text("🔄 Reintentando lectura con IA...")
        texto = await resolver_captcha_con_ia(bytes(image_bytes))
        if texto:
            keyboard = [
                [
                    InlineKeyboardButton("✅ Correcto", callback_data=f"correcto_{texto}"),
                    InlineKeyboardButton("❌ Incorrecto", callback_data=f"incorrecto_{texto}")
                ],
                [InlineKeyboardButton("🔄 Reintentar", callback_data="reintentar")]
            ]
            await query.edit_message_text(
                f"📝 Nueva solución propuesta:\n\n`{texto}`\n\n¿Es correcto?",
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            context.user_data['last_texto'] = texto
        else:
            await query.edit_message_text("❌ Sigo sin poder leer el captcha. Prueba con otra imagen.")

# ================= MAIN =================
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("pause", pause))
    app.add_handler(CommandHandler("resume", resume))
    app.add_handler(CommandHandler("status", status))

    # Manejo de fotos (captchas)
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Botones de respuesta
    app.add_handler(CallbackQueryHandler(button_callback))

    logger.info("🚀 Bot captcha solver iniciado en modo polling")
    app.run_polling()

if __name__ == "__main__":
    main()
