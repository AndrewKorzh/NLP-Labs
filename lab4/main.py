import telebot
import requests
import jsons
from Class_ModelResponse import ModelResponse

API_TOKEN = "..."
bot = telebot.TeleBot(API_TOKEN)
PORT = 1234


user_contexts = {}


@bot.message_handler(commands=["start"])
def send_welcome(message):
    welcome_text = (
        "Hello!\n"
        "Available commands:\n"
        "/start - list of all available commands\n"
        "/model - name of used model\n"
        "/clear - clear context\n"
        "Send me your message and I wil anwser it whith using LLM model."
    )
    bot.reply_to(message, welcome_text)


@bot.message_handler(commands=["model"])
def send_model_name(message):
    response = requests.get(f"http://localhost:{PORT}/v1/models")
    if response.status_code == 200:
        model_info = response.json()
        model_name = model_info["data"][0]["id"]
        bot.reply_to(message, f"Used model: {model_name}")
    else:
        bot.reply_to(message, "Can't get information about model.")


@bot.message_handler(commands=["clear"])
def clear_context(message):
    user_id = message.from_user.id
    user_contexts[user_id] = []
    bot.reply_to(message, "Context was cleared.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    user_query = message.text
    if user_id not in user_contexts:
        user_contexts[user_id] = []
    user_contexts[user_id].append({"role": "user", "content": user_query})

    request = {"messages": user_contexts[user_id]}

    response = requests.post(
        "http://localhost:{PORT}/v1/chat/completions", json=request
    )

    if response.status_code == 200:
        model_response = jsons.loads(response.text, ModelResponse)
        bot_reply = model_response.choices[0].message.content

        user_contexts[user_id].append({"role": "assistant", "content": bot_reply})

        bot.reply_to(message, bot_reply)
    else:
        bot.reply_to(message, "An error occurred when accessing the model.")


if __name__ == "__main__":
    bot.polling(none_stop=True)
