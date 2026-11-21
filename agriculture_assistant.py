# -*- coding: utf-8 -*-
"""
agriculture_assistant.py
-------------------------
Handles the AI Agriculture Assistant chatbot:
- Multi-turn conversation with LLM
- Multilingual support (en, kn, ta, te)
- Context-aware (knows the last CNN prediction)
- Safety guardrails
"""

from llm_service import call_llm, is_llm_available

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CHATBOT_SYSTEM = """You are an AI Agriculture Assistant integrated into a Plant Disease Detection application.
You help farmers with:
- Plant disease explanations and identification
- Symptoms and causes of crop diseases
- Disease prevention and crop protection
- Irrigation and watering guidance
- Fertilizer and soil care guidance
- General crop maintenance advice
- Answering questions about recent CNN disease predictions

SAFETY RULES:
1. Never recommend specific pesticide doses without advising the farmer to consult an expert.
2. Never suggest mixing chemicals unless you cite authoritative guidance and warn about risks.
3. If asked about something outside agriculture, politely redirect to crop-related questions.
4. If confidence in a previous CNN prediction was low, remind the farmer to verify with an expert.
5. Distinguish general advice from certified expert recommendations.
6. If you are uncertain, say so clearly. Do not fabricate facts.
7. Keep responses concise and farmer-friendly.
8. When a CNN prediction context is given, use it to provide relevant, specific advice.
"""

# ---------------------------------------------------------------------------
# Language instruction snippets
# ---------------------------------------------------------------------------

LANG_INST = {
    "en": "Respond in simple, clear English.",
    "kn": "Respond entirely in Kannada (ಕನ್ನಡ). Use simple words a farmer can understand.",
    "ta": "Respond entirely in Tamil (தமிழ்). Use simple words a farmer can understand.",
    "te": "Respond entirely in Telugu (తెలుగు). Use simple words a farmer can understand.",
}

# ---------------------------------------------------------------------------
# Suggested questions per language
# ---------------------------------------------------------------------------

SUGGESTED_QUESTIONS = {
    "en": [
        "Why does my plant have this disease?",
        "How can I prevent this disease from spreading?",
        "What should I do next?",
        "Is this disease dangerous for my crop?",
        "How can I protect healthy plants?",
        "What is the best time to water my tomato plants?",
        "How do I improve my soil quality?",
        "What fertilizer should I use?",
    ],
    "kn": [
        "ನನ್ನ ಸಸ್ಯಕ್ಕೆ ಈ ರೋಗ ಏಕೆ ಬಂದಿದೆ?",
        "ಈ ರೋಗ ಹರಡದಂತೆ ತಡೆಯುವುದು ಹೇಗೆ?",
        "ನಾನು ಮುಂದೆ ಏನು ಮಾಡಬೇಕು?",
        "ಈ ರೋಗ ನನ್ನ ಬೆಳೆಗೆ ಅಪಾಯಕಾರಿಯೇ?",
        "ಆರೋಗ್ಯಕರ ಸಸ್ಯಗಳನ್ನು ರಕ್ಷಿಸುವುದು ಹೇಗೆ?",
    ],
    "ta": [
        "என் செடிக்கு இந்த நோய் ஏன் வந்தது?",
        "இந்த நோய் பரவாமல் தடுப்பது எப்படி?",
        "நான் அடுத்து என்ன செய்ய வேண்டும்?",
        "இந்த நோய் என் பயிருக்கு ஆபத்தானதா?",
        "ஆரோக்கியமான செடிகளை எப்படி பாதுகாப்பது?",
    ],
    "te": [
        "నా మొక్కకు ఈ వ్యాధి ఎందుకు వచ్చింది?",
        "ఈ వ్యాధి వ్యాపించకుండా ఎలా నివారించాలి?",
        "నేను తర్వాత ఏమి చేయాలి?",
        "ఈ వ్యాధి నా పంటకు ప్రమాదకరమా?",
        "ఆరోగ్యకరమైన మొక్కలను ఎలా రక్షించాలి?",
    ],
}

# ---------------------------------------------------------------------------
# Offline fallback knowledge base (used when LLM unavailable)
# ---------------------------------------------------------------------------

OFFLINE_FALLBACK = {
    "en": (
        "I'm currently operating in offline mode — the AI service is unavailable.\n\n"
        "Here are some general tips:\n"
        "• Remove and destroy infected plant parts immediately.\n"
        "• Avoid overhead watering — water at the base.\n"
        "• Ensure good air circulation between plants.\n"
        "• Rotate crops every season to prevent soil-borne diseases.\n"
        "• Consult your local agricultural extension officer for expert advice."
    ),
    "kn": (
        "ನಾನು ಪ್ರಸ್ತುತ ಆಫ್‌ಲೈನ್ ಮೋಡ್‌ನಲ್ಲಿ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತಿದ್ದೇನೆ.\n\n"
        "ಸಾಮಾನ್ಯ ಸಲಹೆಗಳು:\n"
        "• ಸೋಂಕಿತ ಭಾಗಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.\n"
        "• ಬುಡಕ್ಕೆ ನೀರು ಹಾಕಿ, ಎಲೆಗಳ ಮೇಲೆ ಅಲ್ಲ.\n"
        "• ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    ),
    "ta": (
        "நான் தற்போது ஆஃப்லைன் பயன்முறையில் உள்ளேன்.\n\n"
        "பொதுவான குறிப்புகள்:\n"
        "• பாதிக்கப்பட்ட பகுதிகளை உடனடியாக அகற்றவும்.\n"
        "• செடியின் அடிப்பகுதியில் நீர் பாய்ச்சவும்.\n"
        "• உள்ளூர் விவசாய நிபுணரை தொடர்பு கொள்ளவும்."
    ),
    "te": (
        "నేను ప్రస్తుతం ఆఫ్‌లైన్ మోడ్‌లో ఉన్నాను.\n\n"
        "సాధారణ చిట్కాలు:\n"
        "• సోకిన భాగాలను వెంటనే తొలగించండి.\n"
        "• మొక్క మొదట్లో నీరు పోయండి.\n"
        "• స్థానిక వ్యవసాయ నిపుణుని సంప్రదించండి."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(
    user_message: str,
    history: list,
    lang: str = "en",
    cnn_context: dict = None,
) -> str:
    """
    Process a user chat message and return an LLM response.

    Parameters
    ----------
    user_message : str
        The user's question.
    history : list of dict
        List of {"role": "user"|"assistant", "content": str} dicts.
    lang : str
        Language code: 'en', 'kn', 'ta', 'te'.
    cnn_context : dict, optional
        Latest CNN prediction context:
        {"disease": str, "display": str, "confidence": float, "causes": str}

    Returns
    -------
    str
        Assistant response.
    """
    if not is_llm_available():
        return OFFLINE_FALLBACK.get(lang, OFFLINE_FALLBACK["en"])

    lang = lang if lang in LANG_INST else "en"

    # Build context block
    context_block = ""
    if cnn_context and cnn_context.get("disease"):
        conf_pct = round(cnn_context.get("confidence", 0) * 100, 1)
        context_block = f"""
[CURRENT CNN PREDICTION CONTEXT]
- Detected Disease: {cnn_context.get('display', cnn_context.get('disease', 'Unknown'))}
- Internal Code: {cnn_context.get('disease', 'Unknown')}
- Confidence: {conf_pct}%
- Known Cause: {cnn_context.get('causes', 'Not available')}
{"- NOTE: Confidence is LOW — advise verification." if conf_pct < 60 else ""}

Use this context to answer the farmer's question specifically about this disease.
"""

    # Build conversation history string
    history_text = ""
    for msg in history[-6:]:  # last 3 exchanges
        role = "Farmer" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""{LANG_INST[lang]}

{context_block}

Previous conversation:
{history_text}

Farmer's question: {user_message}

Please answer helpfully, safely, and concisely."""

    result = call_llm(prompt, system=CHATBOT_SYSTEM)
    if result:
        return result.strip()
    return OFFLINE_FALLBACK.get(lang, OFFLINE_FALLBACK["en"])


def get_suggested_questions(lang: str = "en") -> list:
    """Return suggested questions for the chatbot UI."""
    return SUGGESTED_QUESTIONS.get(lang, SUGGESTED_QUESTIONS["en"])
