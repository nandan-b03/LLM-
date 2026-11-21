# -*- coding: utf-8 -*-
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether

output_pdf = r"c:\Users\nanda\leaf_disease\leaf_disease\Research_Paper_Additions.pdf"

doc = SimpleDocTemplate(
    output_pdf,
    pagesize=A4,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=14,
    leading=18,
    textColor=colors.HexColor('#1b4332'),
    alignment=1, # Center
    spaceAfter=6
)

subtitle_style = ParagraphStyle(
    'DocSubTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=10,
    leading=14,
    textColor=colors.HexColor('#2d6a4f'),
    alignment=1,
    spaceAfter=14
)

h1_style = ParagraphStyle(
    'H1Style',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=11.5,
    leading=15,
    textColor=colors.HexColor('#1b4332'),
    spaceBefore=12,
    spaceAfter=4,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    'H2Style',
    parent=styles['Heading3'],
    fontName='Helvetica-Bold',
    fontSize=10,
    leading=13,
    textColor=colors.HexColor('#2d6a4f'),
    spaceBefore=8,
    spaceAfter=3,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'BodyStyle',
    parent=styles['BodyText'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor('#222222'),
    spaceAfter=4,
    alignment=4 # Justify
)

bullet_style = ParagraphStyle(
    'BulletStyle',
    parent=body_style,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=3
)

box_style = ParagraphStyle(
    'BoxStyle',
    parent=body_style,
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor('#1b4332')
)

table_text = ParagraphStyle(
    'TableText',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7.8,
    leading=10.5,
    textColor=colors.HexColor('#222222')
)

table_header = ParagraphStyle(
    'TableHeader',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=8,
    leading=11,
    textColor=colors.HexColor('#1b4332')
)

code_style = ParagraphStyle(
    'CodeStyle',
    parent=styles['Normal'],
    fontName='Courier',
    fontSize=7.5,
    leading=10,
    textColor=colors.HexColor('#081c15')
)

story = []

# Title & Subtitle
story.append(Paragraph("A Hybrid Dual-Intelligence Framework: Coupling Convolutional Neural Networks (CNN) with Context-Aware Large Language Models (LLM) for Explainable and Multilingual Agro-Advisory", title_style))
story.append(Paragraph("Integration Methodology, Mathematical Formulation, and Algorithmic Guardrails for Plant Pathology", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2d6a4f'), spaceAfter=10))

# 1. Research Gaps
story.append(Paragraph("1. Research Gaps and Core Contributions", h1_style))
story.append(Paragraph("Conventional automated plant disease detection relies predominantly on deep Convolutional Neural Networks (CNNs). While effective at visual feature classification, standalone CNN models exhibit critical limitations in operational agricultural settings:", body_style))
story.append(Paragraph("• <b>The Black-Box Diagnostic Deficit:</b> Vision classifiers output discrete categorical labels or static templates without pathophysiological explanations or actionable severity details.", bullet_style))
story.append(Paragraph("• <b>Absence of Conversational Context:</b> Farmers cannot ask follow-up questions, request customized preventative schedules, or clarify treatment steps.", bullet_style))
story.append(Paragraph("• <b>Uncalibrated Uncertainty Communication:</b> Predictions are rendered without active uncertainty quantification, presenting severe economic risks if low-confidence results are followed blindly.", bullet_style))
story.append(Paragraph("• <b>Linguistic Exclusivity:</b> Most diagnostic software operates exclusively in English, excluding key regional farming demographics.", bullet_style))
story.append(Spacer(1, 4))

# Callout box
box_data = [[Paragraph("<b>Key Contribution:</b> We introduce a dual-intelligence architecture coupling a 5-layer CNN vision classifier with a generative LLM reasoning engine (Gemini Flash). The vision model provides high-speed visual identification, while the LLM provides context-aware 7-point clinical explanations, multi-turn follow-up dialogue, localized multilingual translations (Kannada, Tamil, Telugu, English), and voice synthesis.", box_style)]]
t_box = Table(box_data, colWidths=[515])
t_box.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0fdf4')),
    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#86efac')),
    ('LINELEFT', (0,0), (0,0), 3.5, colors.HexColor('#16a34a')),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
]))
story.append(t_box)
story.append(Spacer(1, 6))

# 2. Mathematical Formulation
story.append(Paragraph("2. Mathematical and Algorithmic Formulation", h1_style))
story.append(Paragraph("The dual-intelligence pipeline is structured across three formal mathematical phases:", body_style))

story.append(Paragraph("<b>Phase I: Visual Feature Classification and Softmax Distribution</b>", h2_style))
story.append(Paragraph("Given an input leaf tensor <i>I &isin; &real;<sup>50&times;50&times;3</sup></i>, the CNN model <i>f<sub>W</sub>(I)</i> computes logit vector <i>z</i> and softmax probability distribution:", body_style))
story.append(Paragraph("<i>z = f<sub>W</sub>(I) &isin; &real;<sup>K</sup>, &nbsp;&nbsp;&nbsp;&nbsp; P(y = k | I) = e<sup>z_k</sup> / &sum;<sub>j=1</sub><sup>K</sup> e<sup>z_j</sup></i>", bullet_style))
story.append(Paragraph("The predicted label <i>&ycirc;</i> and associated confidence score <i>c</i> are determined by:", body_style))
story.append(Paragraph("<i>&ycirc; = argmax<sub>k &isin; C</sub> P(y = k | I), &nbsp;&nbsp;&nbsp;&nbsp; c = max<sub>k &isin; C</sub> P(y = k | I)</i>", bullet_style))

story.append(Paragraph("<b>Phase II: Diagnostic State Transformation and Uncertainty Guardrail</b>", h2_style))
story.append(Paragraph("The diagnostic state vector <i>S</i> serializes visual classification outputs with active safety constraints and user language <i>L &isin; {en, kn, ta, te}</i>:", body_style))
story.append(Paragraph("<i>S = &lang; &ycirc;, c, Crop(&ycirc;), SeverityFlag(c) &rang;</i>", bullet_style))
story.append(Paragraph("where <i>SeverityFlag(c)</i> injects an uncertainty verification mandate:", body_style))
story.append(Paragraph("<i>SeverityFlag(c) = CONFIRMED if c &ge; 0.60 else LOW_CONFIDENCE_VERIFICATION_REQUIRED</i>", bullet_style))

story.append(Paragraph("<b>Phase III: Conditional Contextual Generative Dialogue</b>", h2_style))
story.append(Paragraph("For dialogue turn <i>t</i> with query <i>q<sub>t</sub></i> and history <i>H<sub>t-1</sub></i>, the LLM generates response <i>r<sub>t</sub></i> conditional on safety prompt &Omega;:", body_style))
story.append(Paragraph("<i>r<sub>t</sub> = argmax<sub>y</sub> &prod;<sub>i=1</sub><sup>M</sup> P(y<sub>i</sub> | y<sub>&lt;i</sub>, &Phi;(S, H<sub>t-1</sub>, &Omega;, L, q<sub>t</sub>))</i>", bullet_style))
story.append(Spacer(1, 6))

# 3. Modular Implementation
story.append(Paragraph("3. Modular Implementation Architecture", h1_style))
story.append(Paragraph("The system was implemented into modular components preserving the original trained CNN model without mutation:", body_style))

table_data = [
    [Paragraph("Module", table_header), Paragraph("Component File", table_header), Paragraph("Functional Responsibility", table_header)],
    [Paragraph("<b>LLM Service Abstraction</b>", table_text), Paragraph("<code>llm_service.py</code>", table_text), Paragraph("Zero-dependency HTTP socket communication via standard <code>urllib.request</code>. Eliminates OpenSSL/urllib3 compatibility faults in scientific Python stacks; provides automatic multi-model failover (Gemini Flash).", table_text)],
    [Paragraph("<b>Clinical Diagnostic Engine</b>", table_text), Paragraph("<code>disease_explanation.py</code>", table_text), Paragraph("Synthesizes a structured 7-point clinical diagnostic report across disease overview, causes, severity, prevention, treatment, precautions, and specialist escalation thresholds. Automatically flags confidence &lt; 60%.", table_text)],
    [Paragraph("<b>Conversational Agro-Agent</b>", table_text), Paragraph("<code>agriculture_assistant.py</code>", table_text), Paragraph("Manages multi-turn dialogue memory, dynamic prompt construction, context inheritance from recent CNN scans, localized question suggestions, and offline fallback knowledge base.", table_text)],
    [Paragraph("<b>Multilingual & Speech UI</b>", table_text), Paragraph("<code>assistant.html</code><br/><code>app.py</code>", table_text), Paragraph("Integrates bidirectional Web Speech ASR (recognition) and neural TTS audio via server proxy <code>/tts</code> supporting English, Kannada, Tamil, and Telugu scripts.", table_text)]
]

t_mod = Table(table_data, colWidths=[110, 115, 290])
t_mod.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8f5e9')),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#b7e4c7')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('RIGHTPADDING', (0,0), (-1,-1), 6),
]))
story.append(t_mod)
story.append(Spacer(1, 8))

# 4. Comparative Evaluation
story.append(Paragraph("4. Comparative Analysis: Baseline vs. Proposed Hybrid System", h1_style))

comp_data = [
    [Paragraph("Evaluation Dimension", table_header), Paragraph("Baseline System (CNN Only)", table_header), Paragraph("Proposed Dual-Intelligence Framework", table_header)],
    [Paragraph("<b>Inference Modality</b>", table_text), Paragraph("Unimodal (Image &rarr; Label Index)", table_text), Paragraph("Multimodal (Image + Textual Dialogue + Voice)", table_text)],
    [Paragraph("<b>Explainability (XAI)</b>", table_text), Paragraph("None (Raw categorical output)", table_text), Paragraph("Comprehensive 7-point structured clinical advisory", table_text)],
    [Paragraph("<b>Interactive Capability</b>", table_text), Paragraph("Static webpage; no follow-up", table_text), Paragraph("Stateful multi-turn conversational AI chatbot", table_text)],
    [Paragraph("<b>Uncertainty Calibration</b>", table_text), Paragraph("Absent; equal display for all confidences", table_text), Paragraph("Active confidence thresholding with &lt;60% warning banner", table_text)],
    [Paragraph("<b>Regional Linguistic Scope</b>", table_text), Paragraph("Hardcoded English snippets", table_text), Paragraph("Localized generation in English, Kannada, Tamil, Telugu", table_text)],
    [Paragraph("<b>Speech Interaction</b>", table_text), Paragraph("Static keyword dictionary lookup", table_text), Paragraph("Bidirectional Voice-to-Text &rarr; LLM &rarr; TTS Speech synthesis", table_text)],
    [Paragraph("<b>Fault Tolerance</b>", table_text), Paragraph("Server crash or empty response on error", table_text), Paragraph("Automated deterministic offline agricultural fallback", table_text)]
]

t_comp = Table(comp_data, colWidths=[120, 185, 210])
t_comp.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8f5e9')),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#b7e4c7')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('RIGHTPADDING', (0,0), (-1,-1), 5),
]))
story.append(t_comp)
story.append(Spacer(1, 8))

# 5. Algorithmic Guardrails
story.append(Paragraph("5. Agricultural Safety & Guardrail Policies", h1_style))
story.append(Paragraph("To ensure reliability and farmer safety, strict guardrail prompts are enforced at runtime:", body_style))
story.append(Paragraph("1. <b>Dosage Restraints:</b> The LLM is strictly prohibited from prescribing chemical pesticide dosages without advising consultation with an authorized local agricultural extension officer.", bullet_style))
story.append(Paragraph("2. <b>Admixture Prohibition:</b> Prohibits recommendations to mix commercial chemicals without verified agronomic protocols.", bullet_style))
story.append(Paragraph("3. <b>Negative Grounding:</b> The model is instructed never to hallucinate disease classifications outside the visual model's findings.", bullet_style))

doc.build(story)
print("PDF generation complete:", output_pdf)
