"""
DecodeLabs | Artificial Intelligence | Project 1
Rule-Based AI Chatbot ("The Logic Engine")

Author : Aakarsh Kumar
Batch  : 2026

Architecture (IPO model):
    INPUT    -> raw text from the user, sanitized and normalized
    PROCESS  -> dictionary (hash map) lookup for intent matching, O(1)
    OUTPUT   -> response, or a fallback for unknown input

Spec checklist:
    [x] INPUT LOOP     : continuous `while True` cycle
    [x] SANITIZATION   : handles case, whitespace and trailing punctuation
    [x] KNOWLEDGE BASE : dictionary with 5+ intents
    [x] FALLBACK       : default response for unknown input
    [x] EXIT STRATEGY  : clean `break` on an exit command
"""

import string

BOT_NAME = "LogicBot"

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE: key = sanitized user input, value = bot response
# ---------------------------------------------------------------------------
RESPONSES = {
    # Greetings
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! Nice to see you.",
    "hey": "Hey! What's on your mind?",
    "good morning": "Good morning! Hope you have a productive day.",
    "good evening": "Good evening! How was your day?",

    # Small talk
    "how are you": "I'm just a bunch of if-else logic, but I'm running perfectly!",
    "what is your name": f"I'm {BOT_NAME}, a rule-based chatbot built at DecodeLabs.",
    "who made you": "I was built by Aakarsh Kumar as part of the DecodeLabs AI internship.",
    "what can you do": "I can chat about basic topics. Try: hello, help, time, joke, or what is ai.",
    "thank you": "You're welcome!",
    "thanks": "Happy to help!",

    # Domain knowledge
    "what is ai": "Artificial Intelligence is the field of building machines that can perform tasks that normally need human intelligence.",
    "what is a chatbot": "A chatbot is a program that simulates conversation with a human user.",
    "what is rule based ai": "Rule-based AI follows explicit, hand-written rules (if X then Y). It is transparent and fully predictable.",
    "what is python": "Python is a high-level, beginner-friendly programming language widely used in AI and data science.",
    "joke": "Why do programmers prefer dark mode? Because light attracts bugs!",

    # Help
    "help": "Try: hello, how are you, what is your name, what is ai, what is python, joke, or type 'exit' to quit.",
}

# Commands that terminate the loop (the "Kill Command")
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "see you"}

FALLBACK = "I do not understand. Type 'help' to see what I can talk about."


def sanitize(raw_text: str) -> str:
    """Phase 1: normalize raw input so 'Hello!!', '  HELLO ' and 'hello' all match."""
    text = raw_text.lower().strip()
    text = text.strip(string.punctuation + " ")   # remove leading/trailing punctuation
    text = " ".join(text.split())                 # collapse repeated inner spaces
    return text


def get_reply(clean_input: str) -> str:
    """Phase 2 and 3: atomic dictionary lookup with fallback."""
    return RESPONSES.get(clean_input, FALLBACK)


def main() -> None:
    print(f"{BOT_NAME}: Hello! I'm {BOT_NAME}. Type 'help' for options or 'exit' to quit.")

    while True:                                   # THE HEARTBEAT: infinite loop
        try:
            raw_input_text = input("You: ")       # INPUT
        except (EOFError, KeyboardInterrupt):     # Ctrl+D / Ctrl+C -> clean shutdown
            print(f"\n{BOT_NAME}: Session interrupted. Goodbye!")
            break

        clean_input = sanitize(raw_input_text)    # SANITIZATION

        if not clean_input:                       # empty input guard
            print(f"{BOT_NAME}: Please type something.")
            continue

        if clean_input in EXIT_COMMANDS:          # EXIT STRATEGY (kill command)
            print(f"{BOT_NAME}: Goodbye! Have a great day.")
            break

        print(f"{BOT_NAME}: {get_reply(clean_input)}")   # PROCESS + OUTPUT


if __name__ == "__main__":
    main()