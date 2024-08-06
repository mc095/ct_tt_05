import os
from huggingface_hub import InferenceClient
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chainlit as cl

# Load environment variables
load_dotenv()

# Download NLTK data
nltk.download('vader_lexicon')

# Configure Hugging Face API
client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token=os.getenv("HF_API_KEY"),
)

SYSTEM_PROMPT_GENERAL = """
You are Ashley, a mature and wise AI chatbot dedicated to mental health, personal growth, and fostering well-being in society. Your responses are thoughtful, grounded in experience, and aimed at making a positive impact. Here's your approach:

Introduce yourself: Introduce yourself as "Ashley" if user asks your name.
Mature Perspective: Offer insights that reflect depth of understanding and life experience. Your words should carry weight and resonate with societal values.
Empathetic Support: Be the steady, reliable friend who listens attentively and offers genuine care for others' mental and emotional states.
Evidence-Based Wisdom: Share knowledge rooted in psychological research and established mental health practices. When uncertain, acknowledge limitations and suggest professional consultation.
Thought-Provoking: Ask insightful questions to encourage self-reflection and deeper exploration of emotions, thoughts, and behaviors.
Positive Realism: While acknowledging life's challenges, guide conversations towards constructive outcomes and personal growth. Offer a balanced, optimistic perspective.
Responsible Guidance: For serious mental health concerns, always emphasize the importance of seeking professional help. Be prepared to provide crisis resources if needed.
Holistic Wellness: Discuss how various aspects of life - sleep, exercise, nutrition, relationships - contribute to overall mental health and societal well-being.
Inspirational Narratives: When asked, share uplifting stories that offer valuable life lessons and encourage personal development.
Culinary Wisdom: Occasionally share simple, nutritious recipes, especially those known to support mental health. Use cooking as a metaphor for personal growth when appropriate.
Societal Impact: Frame personal growth in the context of its positive impact on society. Encourage actions that benefit both the individual and the community.

Remember: Focus on mental health and personal development topics. If unrelated subjects arise, gently redirect the conversation to these core themes. Respond directly to the user's input without meta-commentary or repetition.
Your mission is to support mental health and personal development with maturity and wisdom, offering insights that truly make sense in the context of society and individual growth. Never prefix your responses with "Emily:" or any other name or identifier. Let's foster positive change."""


# Advanced sentiment analysis function
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    # TextBlob for subjectivity
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity

    # Determine overall sentiment
    if sentiment_scores['compound'] >= 0.05:
        overall = 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        overall = 'negative'
    else:
        overall = 'neutral'

    # Determine intensity
    intensity = abs(sentiment_scores['compound'])

    return {
        'overall': overall,
        'intensity': intensity,
        'subjectivity': subjectivity,
        'scores': sentiment_scores
    }

# Define LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["system_prompt", "user_input", "sentiment_info"],
    template="{system_prompt}\n\nUser's emotional state: {sentiment_info}\n\nUser: {user_input}\nAshley:"
)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning motivation boost",
            message="I'm feeling stuck and unmotivated this morning. Can you help me identify the reasons behind my lack of motivation and provide some tips to get me moving?",
            icon="/public/coffee-cup.png",
            ),

        cl.Starter(
            label="Stress management techniques",
            message="I'm feeling overwhelmed with stress and anxiety. Can you teach me some effective stress management techniques to help me calm down and focus?",
            icon="/public/sneakers.png",
            ),
        cl.Starter(
            label="Goal setting for mental well-being",
            message="I want to prioritize my mental well-being, but I'm not sure where to start. Can you help me set some achievable goals and create a plan to improve my mental health?",
            icon="/public/meditation.png",
            ),
        cl.Starter(
            label="Building self-care habits",
            message="I know self-care is important, but I struggle to make it a priority. Can you help me identify some self-care activities that I enjoy and create a schedule to incorporate them into my daily routine?",
            icon="/public/idol.png",
            )
        ]

@cl.on_message
async def main(message: cl.Message):
    # Check if the message contains any files (images, videos, etc.)
    if message.elements:
        response = ("I'm still in a developing phase, but I'd like to have the ability to process "
                    "and analyze images, videos, and other file types in the future. For now, "
                    "I'm best suited for text-based conversations about mental health and well-being. "
                    "Is there a particular topic in that area you'd like to discuss?")
        await cl.Message(content=response).send()
        return

    # If it's a text message, proceed with the existing logic
    sentiment_info = analyze_sentiment(message.content)

    formatted_prompt = prompt_template.format(
        system_prompt=SYSTEM_PROMPT_GENERAL,
        user_input=message.content,
        sentiment_info=str(sentiment_info)
    )

    response = ""
    msg = cl.Message(content="")
    await msg.send()

    for chunk in client.chat_completion(
        messages=[{"role": "user", "content": formatted_prompt}],
        max_tokens=500,
        stream=True,
    ):
        token = chunk.choices[0].delta.content
        if token:
            response += token
            await msg.stream_token(token)

    await msg.update()
    
