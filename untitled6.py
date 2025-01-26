

!pip install transformers
!pip install torch
!pip install nest_asyncio
!pip install python-telegram-bot

import asyncio
import nest_asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import torch
from transformers import pipeline

# STEP 1: Add your Telegram Bot API Token here
API_TOKEN = '7514355786:AAHGtUsuOG35ecJFHVUwEb_sdzNTtRLwrns'  # Replace with your BotFather token

# STEP 2: Set up the pipeline with TinyLlama for conversational responses
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# STEP 3: Define the /start command
async def start(update, context):
    """
    Handles the /start command.
    """
    await update.message.reply_text("Hello! I'm your AI Assistant. How can I help you today?")

# STEP 4: Define the message handler to generate responses
async def handle_message(update, context):
    """
    Handles regular text messages and generates responses using TinyLlama.
    """
    user_message = update.message.text
    print(f"User said: {user_message}")  # Log the user's message

    # Prepare the message in the chat template format
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": user_message},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Get the AI-generated response using TinyLlama model
    try:
        outputs = pipe(prompt, max_new_tokens=190, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        bot_response = outputs[0]["generated_text"]

        # Remove the undesired first few lines from the response
        bot_response_cleaned = bot_response.split("<|assistant|>")[-1].strip()
        await update.message.reply_text(bot_response_cleaned)

    except Exception as e:
        await update.message.reply_text(f"Sorry, I encountered an error: {str(e)}")

# STEP 5: Main function to set up the bot
async def main():
    """
    Sets up and runs the Telegram bot.
    """
    # Initialize the bot application
    application = Application.builder().token(API_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start polling for updates
    print("Bot is running... Press Ctrl+C to stop.")
    await application.run_polling()

# STEP 6: Run the bot in Jupyter/Colab (No need for asyncio.run())
if __name__ == "__main__":
    # Apply nest_asyncio to allow nested event loops in Jupyter/Colab
    nest_asyncio.apply()
    # Simply use the existing event loop
    print("Bot is running in Jupyter Notebook...")
    asyncio.run(main())  # Run the main function using asyncio.run

