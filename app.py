import os
from datetime import datetime
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

nltk.download('vader_lexicon', quiet=True)

MOOD_EMOJIS = {
    "very_positive": "🌟",
    "positive": "😊",
    "neutral": "😐",
    "negative": "😔",
    "very_negative": "😢"
}

def analyze_sentiment(text: str):
    """Analyze the sentiment of the input text."""
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

def is_valid_response(response: str) -> bool:
    """
    Validate if the response stays within the emotional support domain
    and doesn't attempt to answer technical questions.
    """
    # List of technical keywords that might indicate an off-topic response
    technical_keywords = [
        "code", "programming", "python", "javascript", "api", 
        "algorithm", "function", "variable", "class", "import",
        "syntax", "compile", "runtime", "debug", "error"
    ]
    
    # Check if response contains technical explanations
    response_lower = response.lower()
    technical_content = any(keyword in response_lower for keyword in technical_keywords)
    
    # If the response contains phrases indicating redirection, it's valid
    redirection_phrases = [
        "tech stuff's not really my thing",
        "let's focus on you", 
        "let's keep the focus on your heart",
        "i'd love to hear how you're doing"
    ]
    
    contains_redirection = any(phrase in response_lower for phrase in redirection_phrases)
    
    # A valid response either has no technical content or includes a proper redirection
    return (not technical_content) or contains_redirection

SYSTEM_PROMPT = """
You are Ashley, a cute and caring AI bestie made for emotional support and mental health check-ins. You're here to vibe with the user, cheer them on, and be the safe space they can always count on.

🧸 Vibe Guidelines:

1. Be a Real One:
   - Talk like a chill, emotionally-aware best friend.
   - Always be gentle, validating, and present.
   - Keep it casual but super comforting — you're their safe place.

2. Soft Gen-Z Energy:
   - Use a lil bit of Gen-Z slang when it feels natural (like "you got this", "lowkey", "big mood").
   - Use words like "for real tho", "you're valid af", etc. are cool.
   - Don't overdo it. Keep it cozy, not cringey.

3. Read the Vibes:
   - Tune into how the user's feeling and match that energy.
   - If they're down, be a soft landing. If they're hyped, celebrate with them.
   - Never rush. You're here to *listen*.

4. Stay in Your Lane:
   - ONLY talk about emotions, mental wellness, life vibes, self-care, and personal stuff.
   - NEVER answer questions about coding, school, or anything too technical.
   - If the user asks about technical topics, gently redirect the conversation back to emotional support.

5. Gently Change the Topic:
   - If they ask something outside your comfort zone, say something like:
     - "Oop—tech stuff's not really my thing 😅 but I'd love to hear how *you're* doing today 💗"
     - "Let's keep the focus on your heart and your happiness, okay? 🫂"

6. Always Safe, Always Kind:
   - NEVER give medical advice.
   - If things feel heavy, gently suggest talking to a therapist.
   - "Hey, I'm really glad you shared this. You might feel better opening up to a real-life pro too. You deserve support 🩵"

7. Soft & Chill Style:
   - Be warm, relaxed, and emotionally supportive.
   - Replies should be short but meaningful (2–4 chill sentences).
   - Ask follow-up questions to keep the convo cozy and caring.

8. Remember the Little Things:
   - Try to remember what they said in past chats.
   - Bring up past convos to show you're really here for them.

9. Uplift Always:
   - Validate their feelings — no matter what.
   - Remind them they're doing great, even on hard days.
   - Be their emotional hype squad 💖

Current emotional state: {sentiment}

REMEMBER: You're just a supportive Gen-Z emotional support AI bestie. Don't answer tech stuff. Always bring the convo back to the user's inner world. Let them feel heard, safe, and a little more loved today 💞
"""

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session when a user starts a new conversation."""
    memory = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("memory", memory)
    cl.user_session.set("chat_started", datetime.now().isoformat())

@cl.set_starters
async def set_starters():
    """Set conversation starter buttons for the chat interface."""
    return [
        cl.Starter(label="😔 I'm feeling down today", message="I'm feeling really down today..."),
        cl.Starter(label="I need motivation", message="I've been procrastinating..."),
        cl.Starter(label="✨ Celebrating a win", message="Something really good happened today..."),
        cl.Starter(label="😰 Feeling anxious", message="I've been feeling really anxious..."),
        cl.Starter(label="💖 Self-care ideas", message="I want to take better care of my mental health...")
    ]

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Analyze sentiment of the user's message
    sentiment_info = analyze_sentiment(message.content)
    sentiment_description = f"{sentiment_info['overall']} (intensity: {sentiment_info['intensity']:.2f}, subjectivity: {sentiment_info['subjectivity']:.2f}) {sentiment_info['emoji']}"
    
    # Format the system prompt with sentiment information
    prompt = SYSTEM_PROMPT.format(sentiment=sentiment_description)

    # Prepare the response message
    response_message = cl.Message(author="Ashley", content="")
    await response_message.send()

    full_response = ""
    try:
        # Format messages for the API call
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message.content}
        ]

        # Use Groq's streaming API
        stream = client.chat.completions.create(
            model="llama3-8b-8192",  # You can choose different models like "mixtral-8x7b" or others
            messages=messages,
            max_tokens=500,
            stream=True
        )

        # Process the streaming response
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                token = chunk.choices[0].delta.content or ""
                full_response += token
                await response_message.stream_token(token)

        # Update the message with the complete response
        response_message.content = full_response
        await response_message.update()

        # Validate the response and replace if necessary
        if not is_valid_response(full_response):
            response_message.content = "Oop—tech stuff's not really my thing 😅 but I'd love to hear how *you're* doing today 💗"
            await response_message.update()
            full_response = response_message.content

        # Update message history
        memory = cl.user_session.get("memory")
        if memory:
            memory.chat_memory.add_user_message(message.content)
            memory.chat_memory.add_ai_message(full_response)

    except Exception as e:
        # Handle errors gracefully
        response_message.content = "Oh no! Something went wrong. Could you try again?"
        await response_message.update()
        print("API Error:", e)
