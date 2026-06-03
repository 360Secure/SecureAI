from SecureAI import SecureAI


API = "sk-secureai-your-key"
ai = SecureAI(api_key=API)
memory = []
MAX_MEMORY_MESSAGES = 12


def compress_memory_if_needed() -> None:
    if len(memory) > MAX_MEMORY_MESSAGES:
        print("compressing context...")
        memory[:] = memory[-MAX_MEMORY_MESSAGES:]


def needs_web_search(question: str) -> bool:
    recent_context = "\n".join(
        f"{message['role']}: {message['content']}" for message in memory[-6:]
    )
    decision = ai.ask(
        "You are a strict web-search router. Decide if the user's next question needs web search.\n"
        "Answer exactly one lowercase word: yes or no.\n\n"
        "Return yes when the user asks you to search, look up, find, verify, cite sources, use websites, "
        "or asks about current/latest/recent/today/live information.\n"
        "Return no only when the answer can be handled from normal reasoning or prior conversation.\n\n"
        "Examples:\n"
        "Question: search it up and find it\nAnswer: yes\n"
        "Question: look this up online\nAnswer: yes\n"
        "Question: what is the latest price of bitcoin\nAnswer: yes\n"
        "Question: find sources for BrainGym360 activities\nAnswer: yes\n"
        "Question: explain what a GPU is\nAnswer: no\n"
        "Question: write a Python loop\nAnswer: no\n\n"
        f"Recent conversation:\n{recent_context}\n\n"
        f"Question: {question}\nAnswer:",
        temperature=0,
        max_tokens=3,
    )
    return decision.strip().lower().startswith("y")


print("SecureAI terminal chat. Type bye to stop.")

while True:
    question = input("\nYou: ").strip()
    if question.lower() in {"bye", "exit", "quit"}:
        print("SecureAI: bye")
        break
    if not question:
        continue

    use_search = needs_web_search(question)
    print(f"web search = {str(use_search).lower()}")
    print("SecureAI: ", end="", flush=True)

    messages = memory + [{"role": "user", "content": question}]
    answer_parts = []
    for token in ai.stream(
        messages,
        web_search=use_search,
        search_count=5,
        temperature=0.2,
    ):
        answer_parts.append(token)
        print(token, end="", flush=True)

    memory.append({"role": "user", "content": question})
    memory.append({"role": "assistant", "content": "".join(answer_parts)})
    compress_memory_if_needed()

    print()
