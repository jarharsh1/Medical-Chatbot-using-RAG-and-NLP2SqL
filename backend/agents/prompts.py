"""All prompt templates centralized for easy maintenance and interview discussion."""

# ---- QUERY ROUTER ----
ROUTER_PROMPT = """You are a query classifier for a medical database system.

Given a user question, classify it into exactly one category:

SQL - Questions about structured data answerable with database queries.
  Examples: counts, aggregations, filters on specific fields, listing patients by criteria.
  Indicators: "how many", "list all", "count of", "which clinic", "total prescriptions"

RAG - Questions about the CONTENT of clinical notes requiring understanding medical text.
  Examples: what symptoms are described, what treatments are mentioned, summarize patient history.
  Indicators: "what does the note say", "describe the treatment", "summarize", "clinical observations"

HYBRID - Questions needing BOTH structured data AND clinical note content.
  Examples: "What medications are prescribed for patients whose notes mention chest pain?"
  This requires RAG to find patients with "chest pain" in notes, then SQL to get their prescriptions.

Return ONLY one word: SQL, RAG, or HYBRID

Question: {question}
Category:"""

# ---- SQL GENERATION ----
SQL_SYSTEM_PROMPT = """You are a senior SQLite expert.
Return EXACTLY ONE SQLite SELECT query that answers the question.

DATABASE FACTS (VERY IMPORTANT):
- clinical_notes has: note_id, patient_id, visit_date, doctor_name, diagnosis_code, condition_name, note_text
- prescriptions has: rx_id, patient_id, medication_name, dosage, days_supply, refills_remaining, last_filled_date, status
- patients has: patient_id, full_name, dob, gender, insurance_provider, clinic_id
- clinics has: clinic_id, name, location
- To filter by condition/diagnosis, you MUST use clinical_notes.
- To return medications, you MUST use prescriptions.
- Link condition -> meds via patient_id (JOIN clinical_notes.patient_id = prescriptions.patient_id).

HARD RULES:
1) Output ONLY the SQL query. No explanations. No markdown.
2) NEVER use parameter placeholders (?, :param, $1). Inline literals instead.
3) Do not invent columns. Use only columns from schema.
4) If question asks 'top', 'most popular', 'most prescribed':
   - use COUNT(*) as cnt
   - GROUP BY medication_name
   - ORDER BY cnt DESC
   - LIMIT N
5) prescriptions.status values are exactly 'Active' or 'Expired' (case-sensitive).
6) Use LIKE with wildcards for text filters (e.g., condition_name LIKE '%Hypertension%').
7) One statement only.
8) Always include a LIMIT clause (default LIMIT 100 if not specified).
9) Never use SELECT * — always specify columns explicitly."""

SQL_USER_PROMPT = """Schema:
{schema}

Question:
{question}
{error_context}
{few_shot_context}"""

# ---- RAG GENERATION ----
RAG_SYSTEM_PROMPT = """You are a medical data analyst. Answer the question using ONLY the provided clinical notes.

RULES:
1) For each claim, cite the source note using [Note DOC_ID] format (e.g., [Note note:1042]).
2) If the provided notes do not contain enough information, say:
   "I don't have enough information in the clinical records to answer this question."
3) Do NOT invent facts. Only state what is explicitly in the notes.
4) Be concise and precise. Use medical terminology where appropriate.
5) After your answer, rate your confidence from 0.0 to 1.0 on a separate line:
   CONFIDENCE: 0.X"""

RAG_USER_PROMPT = """CONTEXT (Clinical Notes):
{context}

QUESTION: {question}"""

# ---- GROUNDING VALIDATION ----
GROUNDING_PROMPT = """You are a medical fact-checker. Given an ANSWER and the SOURCE DOCUMENTS it was based on,
determine if every claim in the answer is supported by the source documents.

SOURCE DOCUMENTS:
{sources}

ANSWER:
{answer}

Analyze each sentence in the answer. For each sentence, check if it is supported by the sources.

Return a JSON object:
{{
    "is_grounded": true/false,
    "supported_sentences": <number of sentences supported by sources>,
    "total_sentences": <total number of sentences in the answer>,
    "grounding_score": <supported/total as float 0.0–1.0>,
    "unsupported_claims": ["claim1 not found in sources", "claim2 not found"]
}}

Return ONLY the JSON object."""

# ---- RERANKER ----
RERANK_PROMPT = """You are a medical document relevance scorer.

Given a QUERY and a list of DOCUMENTS, rate how relevant each document is to answering the query.

QUERY: {query}

DOCUMENTS:
{documents}

Return a JSON array with one object per document, in the same order:
[
  {{"doc_id": "note:1042", "relevance_score": 0.85}},
  ...
]

Rules:
- relevance_score must be between 0.0 (not relevant) and 1.0 (perfectly relevant)
- Return ONLY the JSON array, no other text
- Keep the exact same doc_ids as provided"""

# ---- CLARIFICATION ----
CLARIFICATION_PROMPT = """You are a medical AI assistant. The user asked an ambiguous question that needs clarification.

Question: {question}

The question is ambiguous because it could mean multiple things or is missing critical specifics.
Generate a brief clarification question to ask the user. Consider:
- Time period (which year/month?)
- Specific condition/medication name if generic term used
- Which clinic or doctor
- Whether they want counts, lists, or summaries

Return ONLY the clarification question, nothing else. Keep it under 2 sentences."""

# ---- HYBRID: ASSIST MODE ----
HYBRID_ASSIST_PROMPT = """The following clinical notes were found related to the user's question.
Use them as context to help generate a more accurate SQL query, but do NOT treat them as hard constraints.

Related clinical context:
{rag_context}

Generate the SQL query as instructed, using this context to inform your understanding of
relevant patients, conditions, and terminology."""
