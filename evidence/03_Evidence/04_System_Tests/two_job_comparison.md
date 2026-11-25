# Two Jobs for the Same Confirmed User

**Profile snapshot:** `student_profile_02` remained byte-for-byte unchanged during both `generate_cv` calls.

| Feature | NLP & Multimodal CV | RAG Agent CV |
|---|---|---|
| Target document_id | `nlp_multimodal_ml_engineer` | `rag_agent_engineer` |
| Summary target | Machine Learning Engineer, NLP & Multimodal AI | AI Engineer |
| First project | Text Classification Study | Coursework Retrieval Assistant |
| Section order | Matched Skills → Experience → Projects → Education → Achievements | Matched Skills → Experience → Projects → Education → Achievements |
| Matched requirements | Python, NLP, transformers, Git | Python, LLMs |
| Missing requirements | Java, AI agents, vector databases, CI/CD, monitoring, Spark, Kafka | specified professional experience, RAG, AI agents, vector databases, cloud platforms |

The NLP CV placed **Text Classification Study** first; the RAG CV placed **Coursework Retrieval Assistant** first. Both used exactly the same confirmed profile object.
