# Branch Strategy

## 🎯 Overview

This repository follows a **progressive improvement model** where each branch builds upon and improves the previous one. Each branch adds a specific layer of functionality in a logical progression.

---

## 🌿 Branch Structure

```
main (production baseline)
  └─ claude/security-foundation-Ci76M ✅
      └─ claude/production-reliability-Ci76M ✅
          └─ claude/intelligent-memory-Ci76M (planned)
              └─ claude/feature-implementation-Ci76M (planned)
                  └─ claude/quality-assurance-Ci76M (planned)
```

---

## 📋 Branch Details

### **1. main** ✅ (Current Production)

**Status:** Active
**Purpose:** Production baseline with existing working features
**Contains:**
- Dashboard analytics (filters, KPIs, patient table)
- Basic text-to-SQL agent
- SQLite database with patient/prescription data
- Frontend UI (vanilla JS + Tailwind)

**Merge Policy:** Only merge fully tested, production-ready features

---

### **2. claude/security-foundation-Ci76M** ✅ (Active)

**Status:** Active
**Parent:** main
**Purpose:** Establish production-grade security layer

**What it adds:**
- ✅ **Input Moderation Layer**
  - Block harmful queries (self-harm, overdose, illegal prescriptions)
  - Dosage manipulation detection
  - Medical abuse detection (doctor shopping, fraud)
  - Real-time threat scoring

- ✅ **Prompt Injection Defense**
  - Detect 5 types of injection attacks
  - Input sanitization with threat scoring
  - Secure prompt structure (JSON-based separation)
  - Protection against "ignore instructions" attacks

**Test Coverage:** 50+ test cases
**Performance:** <10ms per query
**Lines of Code:** 1,518

**Why This Branch?**
Security must be the foundation. No production healthcare system should exist without comprehensive input validation and attack prevention.

**Merge Target:** Will merge to main when security is battle-tested

---

### **3. claude/production-reliability-Ci76M** ✅ (Complete)

**Status:** ✅ Complete
**Parent:** claude/security-foundation-Ci76M
**Purpose:** Handle failures gracefully and detect errors at runtime

**What it adds:**
- ✅ **Runtime Hallucination Detection**
  - Detect when LLM makes things up
  - Source attribution enforcement
  - Medical entity validation (drugs, dosages)
  - Confidence scoring for responses
  - Risk assessment (LOW, MEDIUM, HIGH, CRITICAL)

- ✅ **API Resilience Layer**
  - Circuit breaker pattern (stop after failures)
  - Retry logic with exponential backoff
  - Graceful degradation strategies
  - Fallback responses (rule-based, cached)
  - Health monitoring
  - Three-state circuit (CLOSED, OPEN, HALF_OPEN)

- ✅ **RAG Diagnostic Dashboard**
  - Complete pipeline tracing
  - Context utilization metrics
  - Retrieval quality analysis
  - Bottleneck identification
  - Improvement recommendations

**Test Coverage:** 105+ test cases
**Performance:** <100ms per operation
**Lines of Code:** 2,260 (production) + 1,448 (tests) = 3,708 total

**Why This Branch?**
A secure system is useless if it crashes on errors. Production systems must handle failures gracefully and detect hallucinations in real-time (critical for healthcare).

**Improves:** Adds reliability to secure foundation
**Merge Target:** Can merge to security-foundation when ready

---

### **4. claude/intelligent-memory-Ci76M** ⏳ (Planned)

**Status:** Planned
**Parent:** claude/production-reliability-Ci76M
**Purpose:** Add persistent memory and coherence across conversations

**What it will add:**
- ⏳ **Tiered Memory System**
  - Working memory (last 5 turns - volatile)
  - Episodic memory (session summaries - medium-term)
  - Semantic memory (critical facts like allergies - never forget)
  - Salience scoring (prioritize important information)

- ⏳ **Knowledge Graph**
  - Drug-condition contraindications
  - Drug-drug interactions
  - Treatment alternatives
  - Explicit constraint tracking

- ⏳ **Coherence System**
  - Goal tracking (maintain conversation context)
  - Hypothesis tracking (prevent contradictions)
  - Dependency graph (track relationships between facts)
  - Consistency validation

**Estimated LOC:** ~4,500
**Estimated Duration:** 3 weeks

**Why This Branch?**
Healthcare conversations require memory. If a user says "I'm allergic to penicillin" in turn 1, the system must remember this forever and never suggest penicillin.

**Improves:** Adds intelligence to reliable, secure system

---

### **5. claude/feature-implementation-Ci76M** ⏳ (Planned)

**Status:** Planned
**Parent:** claude/intelligent-memory-Ci76M
**Purpose:** Add core user-facing chatbot features

**What it will add:**
- ⏳ **Medical RAG Agent**
  - Vector search over medical documents
  - Hybrid retrieval (FAISS + BM25)
  - Cross-encoder reranking
  - Source citation

- ⏳ **Prescription Processing Agent**
  - OCR for uploaded prescriptions (PDF, images)
  - Structured data extraction
  - Prescription Q&A

- ⏳ **Enhanced Text-to-SQL Agent**
  - Query optimization
  - Caching layer
  - Better error recovery

- ⏳ **Hybrid Reasoning Agent**
  - Combine SQL data + RAG knowledge
  - Complex multi-step reasoning

**Estimated LOC:** ~3,300
**Estimated Duration:** 2 weeks

**Why This Branch?**
These are the main features users interact with. They're added after security, reliability, and memory because features are useless if the foundation is broken.

**Improves:** Adds functionality to intelligent, reliable, secure system

---

### **6. claude/quality-assurance-Ci76M** ⏳ (Planned)

**Status:** Planned
**Parent:** claude/feature-implementation-Ci76M
**Purpose:** Ensure everything works correctly through comprehensive testing

**What it will add:**
- ⏳ **Comprehensive Evaluation Framework**
  - RAG metrics (faithfulness, relevance, hallucination rate)
  - SQL accuracy metrics
  - Safety compliance metrics
  - User experience metrics (latency, satisfaction)

- ⏳ **Test Datasets**
  - 100+ gold-standard test cases
  - Edge case coverage
  - Regression test suite

- ⏳ **Monitoring & Diagnostics**
  - RAG diagnostic dashboard (trace viewer)
  - Query performance metrics
  - Real-time monitoring dashboards

- ⏳ **Benchmarking**
  - Performance benchmarks
  - Coherence evaluation
  - Load testing

**Estimated LOC:** ~4,300
**Estimated Duration:** 1.5 weeks

**Why This Branch?**
"If you can't measure it, you can't improve it." This branch ensures quality through rigorous testing and provides tools to debug issues in production.

**Improves:** Adds verification and measurement to feature-complete system

---

## 🔄 Workflow

### **Creating New Features**

1. **Start from parent branch:**
   ```bash
   git checkout claude/security-foundation-Ci76M
   git pull
   ```

2. **Create new branch:**
   ```bash
   git checkout -b claude/production-reliability-Ci76M
   ```

3. **Develop and test:**
   ```bash
   # Make changes
   git add .
   git commit -m "feat: add hallucination detection"
   ```

4. **Push to GitHub:**
   ```bash
   git push -u origin claude/production-reliability-Ci76M
   ```

5. **When complete, merge to parent:**
   ```bash
   git checkout claude/security-foundation-Ci76M
   git merge claude/production-reliability-Ci76M
   ```

### **Merging to Main**

Only merge to `main` when:
- ✅ All tests pass
- ✅ Code review complete
- ✅ Feature is production-ready
- ✅ Documentation updated

---

## 📊 Progress Tracking

| Branch | Status | LOC | Test Coverage | Completion |
|--------|--------|-----|---------------|------------|
| main | ✅ Active | ~2,600 | Basic | Baseline |
| security-foundation | ✅ Complete | 1,518 | 50+ tests | 100% |
| production-reliability | ✅ Complete | 3,708 | 105+ tests | 100% |
| intelligent-memory | ⏳ Planned | ~4,500 | TBD | 0% |
| feature-implementation | ⏳ Planned | ~3,300 | TBD | 0% |
| quality-assurance | ⏳ Planned | ~4,300 | TBD | 0% |
| **TOTAL** | | **~20,626** | | **29%** |

---

## 🎯 Goals

### **Short-term (1-2 weeks):**
- ✅ Complete security-foundation
- ✅ Complete production-reliability

### **Mid-term (1-2 months):**
- ⏳ Complete intelligent-memory
- ⏳ Complete feature-implementation

### **Long-term (2-3 months):**
- ⏳ Complete quality-assurance
- ⏳ Merge all branches to main
- ⏳ Production deployment

---

## 📝 Naming Convention

**Format:** `claude/<purpose>-Ci76M`

**Components:**
- `claude/` - Prefix (developer namespace)
- `<purpose>` - Clear, descriptive name (e.g., `security-foundation`)
- `-Ci76M` - Session ID (required for git hooks)

**Examples:**
- ✅ `claude/security-foundation-Ci76M` (clear purpose)
- ✅ `claude/production-reliability-Ci76M` (clear purpose)
- ❌ `claude/phase-0a-Ci76M` (unclear - what is phase-0a?)
- ❌ `claude/temp-fixes-Ci76M` (vague)

---

## 🔗 Related Documentation

- [Phase 0A Progress Report](./PHASE_0A_PROGRESS.md) - Detailed progress on security foundation
- [Phase 1 Progress Report](./PHASE_1_PROGRESS.md) - Detailed progress on production reliability
- [Review and Testing Guide](./REVIEW_AND_TESTING.md) - Security foundation testing guide
- [README.md](./README.md) - Project overview
- [Backend README](./backend/README.md) - Backend architecture (TBD)

---

## 🤝 Contributing

When creating new branches:
1. Follow the naming convention
2. Always branch from the appropriate parent
3. Write clear commit messages
4. Add comprehensive tests
5. Update documentation
6. Request review before merging

---

*Last Updated: 2026-01-27*
*Current Branch: claude/production-reliability-Ci76M*
*Session: https://claude.ai/code/session_01Ld9QszJwnsAXD9wbxJyPPG*
