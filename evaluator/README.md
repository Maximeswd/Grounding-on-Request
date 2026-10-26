# Ground-CF evaluator

A standalone scorer for Ground-CF; no training code is needed. Flip: the answer tracks an edit to the requested source.
Stay: the answer is unchanged when a competing source is edited. JOINT = min(Flip, Stay); Flip and Stay are also
reported on their own, with chance and blind bounds.

    pip install -r evaluator/requirements.txt
    python evaluator/groundcf_eval.py --model <hf-model-id> --split test_id     # teacher-forced likelihood battery
    python evaluator/groundcf_eval.py --generations my_outputs.jsonl           # score your own generations, any model or API

`--generations` parses free-form outputs (failure categories: unmappable, ambiguous) and reports Wilson intervals;
`--model` runs the battery for any model `transformers` can load.
