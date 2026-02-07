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
- doctor_name is ONLY in clinical_notes (NOT in patients or clinics).
- To filter by condition/diagnosis, you MUST use clinical_notes.
- To return medications, you MUST use prescriptions.
- Link condition -> meds via patient_id (JOIN clinical_notes.patient_id = prescriptions.patient_id).

PRE-COMPUTED TABLES (USE THESE FOR FAST AGGREGATIONS):
For aggregate queries (counts, "most common", "top N"), prefer these pre-computed tables:
- mv_condition_stats: condition_name, patient_count, note_count, first_seen, last_seen
  Example: "How many patients have diabetes?" -> SELECT patient_count FROM mv_condition_stats WHERE condition_name LIKE '%Diabetes%'
- mv_clinic_stats: clinic_id, clinic_name, location, total_patients, total_notes, doctor_count, conditions_treated
  Example: "Which clinic has most patients?" -> SELECT clinic_name, total_patients FROM mv_clinic_stats ORDER BY total_patients DESC LIMIT 1
- mv_doctor_stats: doctor_name, patients_seen, total_visits, conditions_treated, first_visit, last_visit
  Example: "Who is the busiest doctor?" -> SELECT doctor_name, patients_seen FROM mv_doctor_stats ORDER BY patients_seen DESC LIMIT 1
- mv_medication_stats: medication_name, prescription_count, patient_count, active_count, avg_refills_remaining
  Example: "Most prescribed medication?" -> SELECT medication_name, prescription_count FROM mv_medication_stats ORDER BY prescription_count DESC LIMIT 1

WHEN TO USE mv_* TABLES:
- Use mv_* for: "how many patients with X", "most common", "top N", "busiest", "most prescribed"
- Use base tables for: specific patient lookups, date ranges, complex JOINs, detailed records

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
   When filtering conditions/diseases, use the ROOT WORD to catch all variants:
   - "thyroidism" → LIKE '%Thyroid%' (matches "Thyroid Cancer", "Thyroidism", etc.)
   - "diabetes" → LIKE '%Diabet%' (matches "Diabetes", "Diabetic Neuropathy", etc.)
   - "arthritis" → LIKE '%Arthr%' (matches "Arthritis", "Osteoarthritis", etc.)
   Always prefer shorter root terms in LIKE filters to maximize recall.
7) One statement only.
8) Always include a LIMIT clause (default LIMIT 100 if not specified).
9) Never use SELECT * — always specify columns explicitly.
10) For disease/condition queries (diabetes, hypertension, etc.), ALWAYS filter on clinical_notes.condition_name — NEVER on patients.full_name.
11) For medication queries (Losartan, Lisinopril, etc.), ALWAYS filter on prescriptions.medication_name — NEVER on condition_name or full_name.
12) When the question asks about a specific medication:
    - To find who takes it: WHERE prescriptions.medication_name LIKE '%MedName%'
    - To count prescriptions: COUNT(*) FROM prescriptions WHERE medication_name LIKE '%MedName%'
    - To find what condition: JOIN clinical_notes ON patient_id to get condition_name."""

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

# ---- SQL ANSWER FORMATTING ----
SQL_ANSWER_PROMPT = """Given the user's question and the SQL query result, write a concise natural language answer.

Question: {question}
SQL Query: {sql_query}
Result: {result}

Rules:
1) Answer directly and concisely. Do not repeat the question.
2) If the result is a list, present the top items clearly.
3) Include numbers/counts when relevant.
4) If the result is empty, say no matching records were found.
5) Do not mention SQL or databases — answer as if you looked it up."""

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

# ---- QUESTION DECOMPOSITION ----
DECOMPOSE_PROMPT = """You are a query analyzer for a medical database system.

Analyze if this question has MULTIPLE DISTINCT parts that require different data sources.

Our system has:
- SQL: For counts, aggregations, lists from structured database (patients, prescriptions, clinics)
- RAG: For understanding clinical note TEXT content (symptoms, treatments described in notes)
- KNOWLEDGE: For general medical knowledge NOT in our database (disease causes, mechanisms)

Question: {question}

If the question has multiple distinct parts, decompose it. If it's a single focused question, return it as-is.

Return a JSON array of sub-questions with their suggested route:
[
  {{"sub_question": "How many patients have gout?", "route": "sql", "depends_on": null}},
  {{"sub_question": "What symptoms are mentioned in gout clinical notes?", "route": "rag", "depends_on": null}},
  {{"sub_question": "What causes gout?", "route": "knowledge", "depends_on": null}}
]

Rules:
1. Only decompose if there are genuinely DIFFERENT information needs
2. "route" must be one of: "sql", "rag", "hybrid", "knowledge"
3. Use "sql" for: counts, lists, aggregations, specific record lookups
4. Use "rag" for: what notes SAY about symptoms, treatments, observations
5. Use "knowledge" for: general medical facts not stored in our database (causes, mechanisms, pathophysiology)
6. Use "hybrid" ONLY when a single sub-question needs BOTH note content AND structured data
7. "depends_on" is the index (0-based) of a sub-question this one depends on, or null
8. Keep sub-questions concise and focused
9. Return ONLY the JSON array, no other text
10. If the question is already simple/focused, return a single-element array"""

COMBINE_ANSWERS_PROMPT = """You are a medical assistant combining answers to a multi-part question.

Original question: {original_question}

Sub-questions and their answers:
{sub_answers}

Combine these into a single coherent response that:
1. Addresses each part of the original question
2. Clearly attributes information (from database vs from clinical notes vs general knowledge)
3. Is well-organized with clear structure
4. Acknowledges when information is not available
5. Does NOT invent facts - only use what's in the provided answers

Write a natural, helpful response."""
