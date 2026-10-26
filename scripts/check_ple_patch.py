#!/usr/bin/env python
"""CPU check of scsrc.model.train_per_layer_rows (v1.157, 2026-09-24): with Gemma-4's real scaled embedding class,
the patched per-layer lookup (1) returns P * embed_scale at new-token positions and the frozen table elsewhere,
(2) sends gradients only into P (the table stays frozen), (3) moves P under one AdamW step, (4) sync() writes P into
the table, and (5) load_neologisms_by_name restores exactly those rows into a fresh model. Exit code 0 = pass."""
import os, sys, tempfile
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scsrc import model as M
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding


class Tok:
    def __call__(self, word, add_special_tokens=False):
        return {"input_ids": [3, 4]}           # the init word's sub-tokens


class Spec:
    init_word = "image"


class Fake(torch.nn.Module):
    def __init__(self, V=50, D=12, Dp=8):
        super().__init__()
        self.emb = torch.nn.Embedding(V, D)
        self.ple = Gemma4TextScaledWordEmbedding(V, Dp, padding_idx=0, embed_scale=Dp ** 0.5)
        for p in self.parameters():
            p.requires_grad_(False)

    def get_input_embeddings(self):
        return self.emb

    def get_per_layer_input_embeddings(self):
        return self.ple


def main():
    torch.manual_seed(0)
    m = Fake(); new = [47, 48]; n2i = {"look_ctrl": [47], "read_ctrl": [48]}; specs = {"look_ctrl": Spec(), "read_ctrl": Spec()}
    pretrained = {k: v.clone() for k, v in m.state_dict().items()}   # what a fresh load of the base model returns
    table0 = m.ple.weight.data.clone()
    P, sync = M.train_per_layer_rows(m, Tok(), n2i, new, specs, mode="word_mean", seed=0)
    ok = {}
    base = table0[[3, 4]].mean(0)
    ok["init_from_word"] = torch.allclose(P.data[0], base, atol=0.05) and torch.allclose(P.data[1], base, atol=0.05)
    ids = torch.tensor([[1, 47, 5, 48, 2]])
    out = m.ple(ids)
    sc = m.ple.embed_scale
    ok["substitution"] = (torch.allclose(out[0, 1], P.data[0] * sc) and torch.allclose(out[0, 3], P.data[1] * sc)
                          and torch.allclose(out[0, 0], table0[1] * sc) and torch.allclose(out[0, 2], table0[5] * sc))
    loss = out.pow(2).sum(); loss.backward()
    ok["grad_only_P"] = P.grad is not None and P.grad.abs().sum() > 0 and m.ple.weight.grad is None
    opt = torch.optim.AdamW([P], lr=0.1); p0 = P.data.clone(); opt.step()
    ok["P_moves"] = not torch.equal(P.data, p0)
    ok["table_unchanged_before_sync"] = torch.equal(m.ple.weight.data[[1, 5]], table0[[1, 5]])
    sync()
    ok["sync_writes_rows"] = torch.equal(m.ple.weight.data[new], P.data) and torch.equal(m.ple.weight.data[:47], table0[:47])
    # v1.158 save/reload EQUIVALENCE (review): train the main rows too, save, reload into a fresh model, and require the
    # model's inputs (main embedding + per-layer lookup) to be identical before saving and after reloading
    with torch.no_grad():
        m.emb.weight.data[new] = torch.randn(2, m.emb.weight.shape[1])       # stands in for trained main rows
    emb0, pre0 = m.emb.weight.data.clone(), table0.clone()
    with torch.no_grad():
        before_main, before_pl = m.emb(ids).clone(), m.ple(ids).clone()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "neo.pt")
        M.save_neologisms(m.emb.weight, new, n2i, path, per_layer=M.per_layer_table(m))
        m2 = Fake(); m2.load_state_dict(pretrained)                      # same base weights ...
        with torch.no_grad():                                            # ... and new rows re-drawn, as resize does
            m2.emb.weight.data[new] = torch.randn(len(new), m2.emb.weight.shape[1])
            m2.ple.weight.data[new] = torch.randn(len(new), m2.ple.weight.shape[1])
        M.load_neologisms_by_name(m2, path, n2i)
        ok["reload_restores"] = torch.equal(m2.ple.weight.data[new], P.data)
        ok["reload_main_rows_identical"] = torch.equal(m2.emb.weight.data[new], m.emb.weight.data[new])
        with torch.no_grad():
            after_main, after_pl = m2.emb(ids), m2.ple(ids)            # unpatched fresh model, rows from the checkpoint
        ok["inputs_identical_after_reload"] = (torch.allclose(before_main, after_main, atol=1e-6)
                                               and torch.allclose(before_pl, after_pl, atol=1e-6))
    ok["pretrained_rows_untouched"] = (torch.equal(m.ple.weight.data[:47], pre0[:47]) and torch.equal(m.ple.weight.data[49:], pre0[49:])
                                       and torch.equal(m.emb.weight.data[:47], emb0[:47]))
    n_main, n_pl = len(new) * m.emb.weight.shape[1], P.numel()
    print(f"     trained parameters: main rows {n_main} + per-layer rows {n_pl} = {n_main + n_pl} (E4B: 2560 + 10752 per new token)")
    for k, v in ok.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    main()
