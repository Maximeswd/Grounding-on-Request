"""Generation helpers shared by diagnostic & evaluate."""
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .collate import _user_content, _encode


class StopOnTokenCount(StoppingCriteria):
    """Stop once `count` occurrences of any id in `ids` have been GENERATED
    (prompt excluded). Lets paired-block outputs run to their final closing
    neologism instead of an arbitrary token budget."""

    def __init__(self, ids, count, prompt_len):
        self.ids = set(int(i) for i in ids)
        self.count = int(count)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids, scores, **kw):
        new = input_ids[0, self.prompt_len:].tolist()
        return sum(1 for t in new if t in self.ids) >= self.count


class StopAfterAnchor(StoppingCriteria):
    """Stop `grace` tokens after the text anchor (default 'final answer:')
    appears in the generated suffix -- the answer finishes, then we cut."""

    def __init__(self, tokenizer, prompt_len, anchor="final answer:", grace=24):
        self.tok = tokenizer
        self.prompt_len = int(prompt_len)
        self.anchor = anchor
        self.grace = int(grace)
        self._seen_at = None

    def __call__(self, input_ids, scores, **kw):
        new = input_ids[0, self.prompt_len:]
        if self._seen_at is None:
            txt = self.tok.decode(new, skip_special_tokens=True).lower()
            if self.anchor in txt:
                self._seen_at = new.shape[0]
            return False
        return (new.shape[0] - self._seen_at) >= self.grace


@torch.no_grad()
def generate(model, processor, image, user_text, device, max_new_tokens=64,
             assistant_prefill: str = "", forbid_ids=None, temperature=0.0,
             stop_token_ids=None, stop_token_count=None, stop_anchor=None,
             anchor_grace=24):
    msgs = [{"role": "user", "content": _user_content(image, user_text)}]
    try:
        enc = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=False).to(device)
    except (TypeError, ValueError):
        enc = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    extra = {k: v for k, v in enc.items()
             if k not in ("input_ids", "attention_mask", "token_type_ids")}
    # optional teacher-forced prefill of the start of the assistant turn
    if assistant_prefill:
        pref = processor.tokenizer(assistant_prefill, add_special_tokens=False,
                                   return_tensors="pt").to(device)
        input_ids = torch.cat([input_ids, pref["input_ids"]], dim=1)
        attn = torch.cat([attn, torch.ones_like(pref["input_ids"])], dim=1)
    proc_list = None
    if forbid_ids:
        from transformers import LogitsProcessorList
        from .model import forbid_ids_processor
        proc_list = LogitsProcessorList([forbid_ids_processor(forbid_ids)])
    stops = []
    if stop_token_ids and stop_token_count:
        stops.append(StopOnTokenCount(stop_token_ids, stop_token_count, input_ids.shape[1]))
    if stop_anchor:
        stops.append(StopAfterAnchor(processor.tokenizer, input_ids.shape[1],
                                     anchor=stop_anchor, grace=anchor_grace))
    gen = model.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=max_new_tokens, do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        logits_processor=proc_list,
        stopping_criteria=StoppingCriteriaList(stops) if stops else None, **extra)
    new = gen[0, input_ids.shape[1]:]
    text = processor.tokenizer.decode(new, skip_special_tokens=False)
    return (assistant_prefill + text).strip()
