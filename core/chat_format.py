"""Shared ChatML-style chat template used across prepare, SFT, and inference.

Vocab layout (specials live at the top of the vocab):
  vocab_size - 3 : <|im_start|>
  vocab_size - 2 : <|im_end|>
  vocab_size - 1 : <|endoftext|>
"""

EOS_TOKEN = '<|endoftext|>'
IM_START = '<|im_start|>'
IM_END = '<|im_end|>'

DEFAULT_SYSTEM = 'You are a helpful assistant.'


def special_token_map(vocab_size):
    return {
        IM_START: vocab_size - 3,
        IM_END: vocab_size - 2,
        EOS_TOKEN: vocab_size - 1,
    }


def render_turn(role, content):
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def render_conversation(messages, add_generation_prompt=False):
    """messages: list of {'role': ..., 'content': ...} dicts."""
    text = ''.join(render_turn(m['role'], m['content']) for m in messages)
    if add_generation_prompt:
        text += f"{IM_START}assistant\n"
    return text
