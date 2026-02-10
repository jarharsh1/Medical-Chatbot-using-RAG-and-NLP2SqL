# Contributing to Medical Chatbot using RAG and NLP2SQL

## Branch Structure

This repository uses a simplified branching strategy:

- **`main`** - Production-ready code (default branch)
- **`claude/*`** - Feature branches created by Claude for specific tasks

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/jarharsh1/Medical-Chatbot-using-RAG-and-NLP2SqL.git
cd Medical-Chatbot-using-RAG-and-NLP2SqL
```

### 2. Set Up Git Identity
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3. Always Start from Main
```bash
git checkout main
git pull origin main
```

### 4. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 5. Make Changes and Push
```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

### 6. Create a Pull Request
- Go to GitHub and create a PR targeting the `main` branch
- Ensure all status checks pass
- Request review from maintainers

## Best Practices

- Keep `main` branch always deployable
- Create small, focused PRs
- Write clear commit messages
- Run tests before pushing
- Delete merged branches

## Current Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production code |
| `claude/production-reliability-Ci76M` | Production reliability improvements |
| `claude/security-foundation-Ci76M` | Security foundation work |
