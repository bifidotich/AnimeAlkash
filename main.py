import datetime
import re
import time
import threading
import queue
import torch
import translators as ts
from utils import *
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, Filters, Updater
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

TOKEN = 'token'
path_img = 'temp/img'
max_dimension = 1000
queue_size_get = 15
queue_size_set = 50
max_requests_people = 3
idx_cuda = 1

queue_get = queue.Queue(maxsize=queue_size_get)
queue_set = queue.Queue(maxsize=queue_size_set)
user_aspect_ratios = {}
lock = threading.Lock()

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "animagine-xl-3.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(f'cuda:{idx_cuda}')


def translate_text(text, target_language='en'):
    return ts.translate_text(text, to_language=target_language, translator='google')


def update_log_on_update(update, text, path_file='temp/logs.txt'):
    if update.message.from_user.username:
        username = update.message.from_user.username
    else:
        username = "ANONIM"
    update_log(f'{username}; {text}', path_file=path_file)


def check_queue_chat(id_chat):
    queue_list = list(queue_get.queue)
    check_list = [p for p in queue_list if p['update'].message.chat_id == id_chat]
    if len(check_list) >= max_requests_people:
        return False
    else:
        return True


def start(update: Update, context: ContextTypes):
    update_log_on_update(update, 'start', path_file='temp/commands.txt')
    start_text = "АлкашТянкаГенератор готов, напиши описание"
    start_text = start_text + "\nДосупные команды: \n/start, /help, /setaspect"
    update.message.reply_text(start_text)
    context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)


def set_aspect(update: Update, context: ContextTypes) -> None:
    update_log_on_update(update, f'setaspect {str(context.args)}', path_file='temp/commands.txt')
    user = update.message.from_user
    if len(context.args) > 0:
        if context.args[0].lower() == 'random':
            if user.id in user_aspect_ratios:
                del user_aspect_ratios[user.id]
                update.message.reply_text("Соотношение сторон сброшено.")
                context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)
                return

        if len(context.args) == 2:
            width, height = context.args
            user_aspect_ratios[user.id] = (int(width), int(height))
            update.message.reply_text(f"Соотношение сторон: {width}:{height}")
            context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)
            return

    text_help = f"\nИспользуйте команду \n/setaspect [ширина] [высота] \nили \n/setaspect random"
    if user.id in user_aspect_ratios:
        ast = user_aspect_ratios[user.id]
        text_help = text_help + f"\nТекущее соотношение сторон: {ast[0]}:{ast[1]}"
        width, height = calculate_dimensions(ast, max_dimension)
        text_help = text_help + f"\nДопустимое разрешение {width}:{height}"
    else:
        text_help = text_help + "\nТекущее соотношение сторон - random"
    update.message.reply_text(text_help)
    context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)


def help_command(update: Update, context: ContextTypes) -> None:
    update_log_on_update(update, 'help', path_file='temp/commands.txt')
    update.message.reply_text("""
    Доступные команды:
    /start
    /setaspect [ширина] [высота] - установить соотношение сторон
    /setaspect random - сбросить соотношение сторон
    /help - вывод подсказки
    """)
    context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)


def get_prompt(update: Update, context: ContextTypes):
    prompt = str(update.message.text)

    if len(prompt) < 3:
        update.message.reply_text("Ты аутист?")
        return
    if len(set(prompt)) < 5:
        update.message.reply_text("С фантазией проблемы?")
        return
    if not check_queue_chat(update.message.chat_id):
        mes = update.message.reply_text(f"Слишком много запросов")
        time.sleep(5)
        context.bot.delete_message(chat_id=mes.chat_id, message_id=mes.message_id)
        context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)
        return
    prompt = translate_text(str(prompt))

    update_log_on_update(update, f'{update.message.text}; {prompt}', path_file='temp/requests.txt')

    if not re.compile(r"^[a-zA-Z0-9.':\",’!?\- ]+$").match(prompt):
        update.message.reply_text("Неверный текст")
        return

    num_que = queue_get.qsize()
    if num_que < queue_size_get:
        message_bot = update.message.reply_text(f"Запрос принят, номер в очереди: {num_que + 1}\nprompt: {prompt}")
        with lock:
            queue_get.put({'update': update, 'prompt': prompt, 'context': context, 'message_bot': message_bot})
    else:
        update.message.reply_text("Очередь переполнена")
        return


def process_model():
    while True:
        time.sleep(1)
        if not queue_get.empty():
            with lock:
                req = queue_get.get()
            if req['update'].message.from_user.id in user_aspect_ratios:
                as_tuple = user_aspect_ratios[req['update'].message.from_user.id]
            else:
                as_tuple = random_proportional_resolution()
            width, height = calculate_dimensions(as_tuple, max_dimension)
            try:
                req['path_photo'] = main_generate(req['prompt'], width, height, return_path=True)
            except Exception as e:
                update_log(f'Ошибка генерации, {str(e)}', path_file='temp/errors.txt')
            try:
                with lock:
                    queue_set.put(req)
            except Exception as e:
                update_log(str(e), path_file='temp/errors.txt')


def process_set():
    while True:
        time.sleep(1)
        if not queue_set.empty():
            with lock:
                req = queue_set.get()
            update = req['update']
            context = req['context']
            message_bot = req['message_bot']
            photo = open(req['path_photo'], 'rb')

            del_bot_chat_id = message_bot.chat_id
            del_bot_message_id = message_bot.message_id

            try:
                update.message.reply_photo(photo=photo, caption=f'prompt: {update.message.text}\ntranslate: {req["prompt"]}')
                context.bot.delete_message(chat_id=update.message.chat_id, message_id=update.message.message_id)
            except Exception as e:
                update_log(str(e), path_file='temp/errors.txt')
                try:
                    req['message_bot'] = update.message.reply_text(f"Ошибка, prompt: {req['prompt']}\nПовторная попытка...")
                except Exception as e:
                    er = f'ошибка отправки отчет, {update.message.chat.id}, {e}'
                    update_log(str(er), path_file='temp/errors.txt')
                with lock:
                    queue_set.put(req)

            context.bot.delete_message(chat_id=del_bot_chat_id, message_id=del_bot_message_id)


def main_generate(prompt, width, height, return_path=False):
    negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"

    results = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=7,
        num_inference_steps=28
    )

    image = results.images[0]
    img_path = f'{path_img}/{random_string(12)}.png'
    track_dir(img_path)
    image.save(img_path)
    if return_path:
        return img_path
    return open(img_path, 'rb')


def main():

    updater = Updater(TOKEN, use_context=True)
    updater.dispatcher.add_handler(CommandHandler("start", start))
    updater.dispatcher.add_handler(CommandHandler("setaspect", set_aspect))
    updater.dispatcher.add_handler(CommandHandler("help", help_command))
    updater.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, get_prompt))

    threading.Thread(target=process_model, daemon=True).start()
    threading.Thread(target=process_set, daemon=True).start()

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
