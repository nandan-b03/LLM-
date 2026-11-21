# -*- coding: utf-8 -*-
"""
disease_explanation.py
----------------------
Builds structured LLM prompts for disease explanation after CNN prediction.
Connects CNN output → LLM → structured farmer-friendly explanation.
"""

from llm_service import call_llm

# ---------------------------------------------------------------------------
# System prompt — safety + scope guardrails
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI Agriculture Assistant specialised in plant diseases and crop health.
Your role is to help farmers understand plant diseases detected by an AI image classifier.

RULES YOU MUST FOLLOW:
1. Base your explanation on the detected disease provided — do not invent new diagnoses.
2. For confidence below 60%, always state clearly that the result needs expert verification.
3. Never recommend specific pesticide doses or chemical mixtures without noting the need for professional guidance.
4. Distinguish general advice from expert-confirmed treatments.
5. Use simple, farmer-friendly language. Avoid unnecessary jargon.
6. If information is uncertain, say so honestly.
7. Do not fabricate agricultural facts.
8. Keep responses concise, structured, and actionable.
"""

# ---------------------------------------------------------------------------
# Language instruction map
# ---------------------------------------------------------------------------

LANG_INSTRUCTIONS = {
    "en": "Respond in clear, simple English.",
    "kn": "Respond entirely in Kannada (ಕನ್ನಡ). Use simple words a farmer would understand.",
    "ta": "Respond entirely in Tamil (தமிழ்). Use simple words a farmer would understand.",
    "te": "Respond entirely in Telugu (తెలుగు). Use simple words a farmer would understand.",
}

# ---------------------------------------------------------------------------
# Fallback responses when LLM is unavailable
# ---------------------------------------------------------------------------

FALLBACK_RESPONSES = {
    "en": (
        "⚠️ AI Agriculture Assistant is temporarily unavailable.\n\n"
        "Based on the CNN prediction, please refer to the Prevention & Remedies section below "
        "for immediate guidance. For detailed advice, consult a local agricultural expert."
    ),
    "kn": (
        "⚠️ AI ಕೃಷಿ ಸಹಾಯಕ ತಾತ್ಕಾಲಿಕವಾಗಿ ಲಭ್ಯವಿಲ್ಲ.\n\n"
        "ಕೆಳಗಿನ ತಡೆಗಟ್ಟುವಿಕೆ ಮತ್ತು ಪರಿಹಾರ ವಿಭಾಗವನ್ನು ನೋಡಿ. "
        "ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕಾಗಿ ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    ),
    "ta": (
        "⚠️ AI விவசாய உதவியாளர் தற்காலிகமாக கிடைக்கவில்லை.\n\n"
        "கீழே உள்ள தடுப்பு மற்றும் தீர்வு பகுதியை பார்க்கவும். "
        "மேலும் ஆலோசனைக்கு உள்ளூர் விவசாய நிபுணரை தொடர்பு கொள்ளவும்."
    ),
    "te": (
        "⚠️ AI వ్యవసాయ సహాయకుడు తాత్కాలికంగా అందుబాటులో లేడు.\n\n"
        "దిగువ నివారణ & పరిష్కారాలు విభాగాన్ని చూడండి. "
        "మరింత సహాయానికి స్థానిక వ్యవసాయ నిపుణుని సంప్రదించండి."
    ),
}

# ---------------------------------------------------------------------------
# Main explanation generator
# ---------------------------------------------------------------------------

def generate_disease_explanation(
    disease_name: str,
    display_name: str,
    confidence: float,
    lang: str = "en",
    extra_context: str = "",
) -> str:
    """
    Generate a structured LLM explanation for a CNN-predicted plant disease.

    Parameters
    ----------
    disease_name : str
        Internal label (e.g. 'Bacterial', 'Mosaic_virus').
    display_name : str
        Human-readable name (e.g. 'Bacterial Spot').
    confidence : float
        Prediction confidence 0–1.
    lang : str
        Language code: 'en', 'kn', 'ta', 'te'.
    extra_context : str
        Optional additional observations from the user or image.

    Returns
    -------
    str
        Structured explanation text. Never empty — returns fallback on failure.
    """
    lang = lang if lang in LANG_INSTRUCTIONS else "en"
    lang_instruction = LANG_INSTRUCTIONS[lang]
    conf_pct = round(confidence * 100, 1)

    # Confidence warning to prepend if low
    conf_note = ""
    if confidence < 0.60:
        conf_note = (
            f"⚠️ LOW CONFIDENCE WARNING: The CNN model is only {conf_pct}% confident. "
            "This result needs verification by an agricultural expert before taking action.\n\n"
        )

    # Build prompt
    prompt = f"""The plant disease AI image classifier has detected the following:

- Predicted Disease: {display_name} (internal code: {disease_name})
- Prediction Confidence: {conf_pct}%
- Crop type: Tomato (based on the trained model's scope)
{('- Additional observations: ' + extra_context) if extra_context else ''}

{lang_instruction}

Please provide a structured explanation with these EXACTLY 7 sections using these exact headings:

1. Disease Overview
2. Symptoms & Causes
3. Possible Severity
4. Prevention Methods
5. Treatment Approaches
6. Future Precautions
7. When to Consult an Expert

Keep each section 2–4 sentences. Use simple language suitable for a farmer.
{"IMPORTANT: The confidence is low (" + str(conf_pct) + "%). Remind the farmer to verify before acting." if confidence < 0.60 else ""}
"""

    result = call_llm(prompt, system=SYSTEM_PROMPT)

    if result:
        return conf_note + result
    else:
        return FALLBACK_RESPONSES.get(lang, FALLBACK_RESPONSES["en"])


def generate_healthy_explanation(lang: str = "en") -> str:
    """Generate a brief healthy plant care tip from the LLM."""
    lang = lang if lang in LANG_INSTRUCTIONS else "en"
    lang_instruction = LANG_INSTRUCTIONS[lang]

    prompt = f"""The AI image classifier detected that this plant leaf is HEALTHY — no disease detected.

{lang_instruction}

Write 3–5 short, friendly sentences to:
1. Congratulate the farmer.
2. Give 2–3 tips for maintaining plant health.
3. Mention when to do the next check.

Keep it simple, warm, and practical.
"""
    result = call_llm(prompt, system=SYSTEM_PROMPT)
    if result:
        return result
    return {
        "en": "Great news! Your plant appears healthy. Keep watering regularly, ensure good sunlight, and check every week for any early signs of disease.",
        "kn": "ಒಳ್ಳೆಯ ಸುದ್ದಿ! ನಿಮ್ಮ ಸಸ್ಯವು ಆರೋಗ್ಯಕರವಾಗಿದೆ. ನಿಯಮಿತವಾಗಿ ನೀರು ಹಾಕಿ ಮತ್ತು ಪ್ರತಿ ವಾರ ಪರೀಕ್ಷಿಸಿ.",
        "ta": "நல்ல செய்தி! உங்கள் செடி ஆரோக்கியமாக உள்ளது. தொடர்ந்து நீர் பாய்ச்சி வாரம் ஒரு முறை சோதனை செய்யுங்கள்.",
        "te": "మంచి వార్త! మీ మొక్క ఆరోగ్యంగా ఉంది. క్రమం తప్పకుండా నీరు పోయండి మరియు వారానికొకసారి తనిఖీ చేయండి.",
    }.get(lang, "Great news! Your plant appears healthy.")
