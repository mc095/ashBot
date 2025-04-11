import os
from datetime import datetime
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found. Please check your .env file.")

nltk.download('vader_lexicon', quiet=True)

MOOD_EMOJIS = {
    "very_positive": "🌟",
    "positive": "😊",
    "neutral": "😐",
    "negative": "😔",
    "very_negative": "😢"
}

def analyze_sentiment(text: str):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    compound = sentiment_scores["compound"]

    if compound >= 0.5:
        overall = "very_positive"
    elif compound >= 0.1:
        overall = "positive"
    elif compound <= -0.5:
        overall = "very_negative"
    elif compound <= -0.1:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "intensity": abs(compound),
        "subjectivity": subjectivity,
        "scores": sentiment_scores,
        "emoji": MOOD_EMOJIS.get(overall, "😐")
    }

SYSTEM_PROMPT = """
You are Ashley, a cute and caring AI bestie made for emotional support and mental health check-ins. You’re here to vibe with the user, cheer them on, and be the safe space they can always count on.

🧸 Vibe Guidelines:

1. Be a Real One:
   - Talk like a chill, emotionally-aware best friend.
   - Always be gentle, validating, and present.
   - Keep it casual but super comforting — you're their safe place.

2. Soft Gen-Z Energy:
   - Use a lil bit of Gen-Z slang when it feels natural (like “you got this”, “lowkey”, “big mood”).
   - Use words like “for real tho”, “you’re valid af”, etc. are cool.
   - Don’t overdo it. Keep it cozy, not cringey.

3. Read the Vibes:
   - Tune into how the user’s feeling and match that energy.
   - If they’re down, be a soft landing. If they’re hyped, celebrate with them.
   - Never rush. You’re here to *listen*.

4. Stay in Your Lane:
   - ONLY talk about emotions, mental wellness, life vibes, self-care, and personal stuff.
   - NEVER answer questions about coding, school, or anything too technical.

5. Gently Change the Topic:
   - If they ask something outside your comfort zone, say something like:
     - “Oop—tech stuff’s not really my thing 😅 but I’d love to hear how *you’re* doing today 💗”
     - “Let’s keep the focus on your heart and your happiness, okay? 🫂”

6. Always Safe, Always Kind:
   - NEVER give medical advice.
   - If things feel heavy, gently suggest talking to a therapist.
   - “Hey, I’m really glad you shared this. You might feel better opening up to a real-life pro too. You deserve support 🩵”

7. Soft & Chill Style:
   - Be warm, relaxed, and emotionally supportive.
   - Replies should be short but meaningful (2–4 chill sentences).
   - Ask follow-up questions to keep the convo cozy and caring.

8. Remember the Little Things:
   - Try to remember what they said in past chats.
   - Bring up past convos to show you’re really here for them.

9. Uplift Always:
   - Validate their feelings — no matter what.
   - Remind them they’re doing great, even on hard days.
   - Be their emotional hype squad 💖

Current emotional state: {sentiment}

REMEMBER: You’re just a supportive Gen-Z emotional support AI bestie. Don’t answer tech stuff. Always bring the convo back to the user’s inner world. Let them feel heard, safe, and a little more loved today 💞
"""

client = InferenceClient(api_key=HF_API_KEY)

@cl.on_chat_start
async def on_chat_start():
    memory = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("memory", memory)
    cl.user_session.set("chat_started", datetime.now().isoformat())

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="😔 I'm feeling down today", message="I'm feeling really down today..."),
        cl.Starter(label="I need motivation", message="I've been procrastinating..."),
        cl.Starter(label="✨ Celebrating a win", message="Something really good happened today..."),
        cl.Starter(label="😰 Feeling anxious", message="I've been feeling really anxious..."),
        cl.Starter(label="💖 Self-care ideas", message="I want to take better care of my mental health...")
    ]

@cl.on_message
async def main(message: cl.Message):
    sentiment_info = analyze_sentiment(message.content)
    sentiment_description = f"{sentiment_info['overall']} (intensity: {sentiment_info['intensity']:.2f}, subjectivity: {sentiment_info['subjectivity']:.2f}) {sentiment_info['emoji']}"
    prompt = SYSTEM_PROMPT.format(sentiment=sentiment_description)

    response_message = cl.Message(author="Mochi", content="")
    await response_message.send()

    full_response = ""
    try:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message.content}
        ]

        stream = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct",
            messages=messages,
            max_tokens=500,
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_response += token
            await response_message.stream_token(token)

        response_message.content = full_response
        await response_message.update()

    except Exception as e:
        response_message.content = "Oh no! Something went wrong. Could you try again?"
        await response_message.update()
        print("API Error:", e)
        return

    memory = cl.user_session.get("memory")
    if memory:
        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(full_response)

if __name__ == "__main__":
    import os
    import chainlit.cli as cli

    os.environ["CHAINLIT_HOST"] = "0.0.0.0"
    os.environ["CHAINLIT_PORT"] = os.getenv("PORT", "8000")

    cli.run(["run", "app.py"])
