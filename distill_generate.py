"""Teacher-distillation data generator.

Uses an OpenAI chat model as the teacher to produce ChatML-ready SFT
conversations for the ~100M student. Two prompt sources:

  --source no_robots   stream human-written prompts from HuggingFaceH4/no_robots
                       and replace the reference answer with the teacher's
  --source synthetic   the teacher invents both the question and the answer,
                       seeded from a built-in topic list (fully synthetic
                       general-knowledge QA, cosmopedia-style but conversational)

Output: one JSONL per conversation, {"messages": [{role, content}, ...]} with a
fixed system prompt (chat_format.DEFAULT_SYSTEM) so the data lands exactly on
the distribution the chat server uses. Pack it into SFT shards with:

  python3 sft_prepare.py --input-jsonl data_cache/distill/teacher_sft.jsonl

API key resolution order for --api-key-env (default OPENAI_API_KEY): process
environment, then .env.local at the repo root. The key is never printed.

Example:
  python3 distill_generate.py --source no_robots --max-examples 2000
  python3 distill_generate.py --source synthetic --max-examples 2000
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from chat_format import DEFAULT_SYSTEM

# Keep teacher answers short and plain: a ~100M student imitates surface form
# long before it absorbs content, so targets should be simple and declarative.
TEACHER_INSTRUCTION = (
    "You are writing training answers for a very small language model. "
    "Answer accurately, clearly, and concisely: 1-4 short sentences of plain "
    "everyday language, no markdown, no lists, no jargon. If the question is "
    "ambiguous, answer the most likely reading. Never refuse a benign question."
)

SYNTHETIC_INSTRUCTION = (
    "Invent one simple factual question about the topic below that a curious "
    "10-year-old might ask, then answer it accurately in 1-3 short sentences of "
    "plain language. Reply in exactly this format (two lines, no markdown):\n"
    "Q: <question>\n"
    "A: <answer>"
)

SEED_TOPICS = [
    "animals", "space and planets", "the human body", "oceans", "weather",
    "dinosaurs", "plants and trees", "insects", "birds", "volcanoes",
    "famous inventors", "ancient Egypt", "the Roman Empire", "world geography",
    "rivers and mountains", "food and cooking", "sports", "music instruments",
    "famous painters", "everyday machines", "electricity", "computers",
    "the internet", "trains and airplanes", "ships", "bridges and buildings",
    "mathematics", "shapes and numbers", "money and trade", "farms",
    "forests", "deserts", "the water cycle", "seasons", "the Moon",
    "the Sun and stars", "gravity", "magnets", "colors and light",
    "sound", "senses", "sleep and dreams", "germs and health", "teeth",
    "books and stories", "languages", "maps", "clocks and calendars",
]


def load_api_key(env_name):
    """Resolve the API key without ever printing it: env first, then .env.local."""
    if os.environ.get(env_name):
        return os.environ[env_name]
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env.local')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == env_name:
                    return v.strip().strip('"').strip("'")
    raise SystemExit(f"error: {env_name} not found in environment or .env.local")


def ask_teacher(client, model, messages, max_tokens, temperature):
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature)
    usage = resp.usage
    return (resp.choices[0].message.content.strip(),
            (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0))


def iter_no_robots_prompts():
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/no_robots', split='train', streaming=True)
    for row in ds:
        user = next((m for m in row['messages'] if m['role'] == 'user'), None)
        if user and user['content'].strip():
            yield user['content'].strip()


def parse_synthetic_qa(text):
    """Parse the 'Q: ...\\nA: ...' format. Returns (question, answer) or None."""
    text = text.strip()
    if not text.startswith('Q:') or '\nA:' not in text:
        return None
    q_part, a_part = text[2:].split('\nA:', 1)
    q, a = q_part.strip(), a_part.strip()
    return (q, a) if q and a else None


def build_example(client, args, task):
    """One teacher call -> messages dict + token usage. task is a raw prompt
    (no_robots) or a topic string (synthetic)."""
    if args.source == 'no_robots':
        answer, usage = ask_teacher(
            client, args.model,
            [{'role': 'system', 'content': TEACHER_INSTRUCTION},
             {'role': 'user', 'content': task}],
            args.max_tokens, args.temperature)
        return {'messages': [{'role': 'system', 'content': DEFAULT_SYSTEM},
                             {'role': 'user', 'content': task},
                             {'role': 'assistant', 'content': answer}]}, usage
    raw, usage = ask_teacher(
        client, args.model,
        [{'role': 'system', 'content': SYNTHETIC_INSTRUCTION},
         {'role': 'user', 'content': f"Topic: {task}"}],
        args.max_tokens, args.temperature)
    qa = parse_synthetic_qa(raw)
    if qa is None:
        raise ValueError(f"unparseable synthetic QA: {raw[:120]!r}")
    q, a = qa
    return {'messages': [{'role': 'system', 'content': DEFAULT_SYSTEM},
                         {'role': 'user', 'content': q},
                         {'role': 'assistant', 'content': a}]}, usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['no_robots', 'synthetic'],
                        default='no_robots')
    parser.add_argument('--model', default=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
                        help='teacher model (env OPENAI_MODEL overrides the default)')
    parser.add_argument('--max-examples', type=int, default=1000)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY',
                        help='name of the env var holding the OpenAI API key')
    parser.add_argument('--output', default=None,
                        help='default: data_cache/distill/teacher_<source>.jsonl')
    parser.add_argument('--resume', action='store_true',
                        help='append to an existing output file, skipping the '
                             'number of tasks it already contains')
    args = parser.parse_args()

    output = args.output or os.path.join(
        'data_cache', 'distill', f'teacher_{args.source}.jsonl')
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if args.source == 'no_robots':
        tasks = []
        for prompt in iter_no_robots_prompts():
            tasks.append(prompt)
            if len(tasks) >= args.max_examples:
                break
    else:
        rng = random.Random(args.seed)
        tasks = [rng.choice(SEED_TOPICS) for _ in range(args.max_examples)]

    skip = 0
    if args.resume and os.path.exists(output):
        with open(output) as f:
            skip = sum(1 for line in f if line.strip())
        print(f"resume: skipping {skip} already-generated examples")
    tasks = tasks[skip:]
    if not tasks:
        print("nothing to do")
        return

    from openai import OpenAI
    client = OpenAI(api_key=load_api_key(args.api_key_env))

    done = failed = 0
    prompt_tokens = completion_tokens = 0
    with open(output, 'a') as out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(build_example, client, args, t): i
                       for i, t in enumerate(tasks)}
            for fut in as_completed(futures):
                try:
                    example, usage = fut.result()
                except Exception as e:  # noqa: BLE001 — keep going past bad samples
                    failed += 1
                    print(f"  task {futures[fut]} failed: {e}", file=sys.stderr)
                    continue
                out.write(json.dumps(example, ensure_ascii=False) + '\n')
                out.flush()
                prompt_tokens += usage[0]
                completion_tokens += usage[1]
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(tasks)} examples "
                          f"({failed} failed, {completion_tokens:,} completion tokens)")

    print(f"wrote {done} examples ({failed} failed) to {output}")
    print(f"teacher usage: {prompt_tokens:,} prompt + {completion_tokens:,} "
          f"completion tokens")


if __name__ == '__main__':
    main()
