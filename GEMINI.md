Project Context: JR RAG Job Application Assistant
1. Project Goal
Build a local, agentic RAG application to automate the creation of tailored, 2-page CVs from a "Master CV" dataset. The system simulates a recruiter's evaluation process to ensure high-relevancy matches.

2. Technical Stack
IDE: Antigravity (VS Code Fork)

Backend: Python

Frontend: React (transitioning from basic UI to an interactive Preview/Edit dashboard)

Vector DB: FAISS (Local persistence)

Models: Gemini 3 Pro / Flash

3. Data Architecture (Crucial)
Source Data: A "Master CV" containing 25+ years of Senior Engineering Leadership experience (primarily at Ericsson Canada).

Chunking Strategy: Hierarchical and bullet-based.

Metadata Injection: Every chunk is prepended with its hierarchy to maintain context during retrieval:
[Section] | [Company/Sub-heading] | [Year Span] | [Achievement Bullet]

4. Current Pipeline Logic
Ingestion: Python scripts parse the Master CV into the hierarchical bullet format and store them in FAISS.

Retrieval: The system compares a Job Description (JD) against the FAISS index.

Expansion: Instead of 5-6 paragraphs, the system retrieves the top 30-40 most relevant bullets.

Review (In Progress): The React frontend will display these bullets for human-in-the-loop "Veto/Approval" before drafting.

Generation: The approved bullets are sent to Gemini to be synthesized into a professional, cohesive 2-page Markdown CV.

5. Development Priorities
UI/UX: Build a "Selection Checklist" in React to review retrieved bullets.

Tone Control: Ability to toggle between "Strategic Leadership" and "Hands-on Technical" focus depending on the JD.

Formatting: Ensuring the final output adheres to executive standards (PDF export).