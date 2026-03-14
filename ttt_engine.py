"""
ttt_engine.py — Test-Time Training engine.

Supports both:
  A) RetentionLabs TTT models (TTT-Linear / TTT-MLP) — native TTT layers
  B) Standard HuggingFace models (granite, Qwen) — surgical DualMLP retrofit

Auto-detects which approach to use based on model architecture.
"""

import math
import time
import pickle
import io
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16

# Inner-loop hyperparameters
# Kept very low to prevent the reconstruction loss from collapsing to
# near-zero in a single step, which blows up the dynamic path weights
# and corrupts generation on the next turn.
INNER_LR_MLP   = 1e-6    # was 1e-5
INNER_LR_LINEAR= 1e-5    # was 1e-4
MAX_GRAD_NORM  = 1.0
CHUNK_TOKENS   = 64
MINI_BATCH     = 16

# PonderTTT gating
EMA_ALPHA      = 0.1
EMA_THRESHOLD  = 0.85

# Selective layer freezing (standard transformer retrofit)
FREEZE_FRACTION = 0.75

# Generation
MAX_NEW_TOKENS  = 150    # was 400 — keep responses concise
TEMPERATURE     = 0.75
TOP_P           = 0.9
REPETITION_PENALTY = 1.05


# ─────────────────────────────────────────────────────────────
# DUAL-MLP TTT HEAD
# Used when retrofitting a standard transformer model
# ─────────────────────────────────────────────────────────────

class DualMLPTTTHead(nn.Module):
    """
    Replaces a transformer MLP block with:
      - Static path (frozen original MLP)  → preserves pre-trained knowledge
      - Dynamic path (trainable TTT MLP)   → adapts to conversation

    Output = static_out + LayerNorm(dynamic_out)

    IMPORTANT: LayerNorm is applied only to the dynamic delta, NOT to
    static_out + dynamic_out. Wrapping the sum would re-normalize the
    residual stream and corrupt the base model's output from turn one.
    """

    def __init__(self, original_mlp: nn.Module, hidden_size: int):
        super().__init__()

        self.static_mlp = original_mlp
        for p in self.static_mlp.parameters():
            p.requires_grad = False

        # Dynamic path — SwiGLU gated MLP
        inter = hidden_size * 2
        self.dynamic_up   = nn.Linear(hidden_size, inter, bias=False)
        self.dynamic_gate = nn.Linear(hidden_size, inter, bias=False)
        self.dynamic_down = nn.Linear(inter, hidden_size, bias=False)
        self.norm         = nn.LayerNorm(hidden_size)

        # Reconstruction projections for PonderTTT loss
        proj = hidden_size // 4
        self.key_proj = nn.Linear(hidden_size, proj, bias=False)
        self.val_proj = nn.Linear(hidden_size, proj, bias=False)
        self.val_pred = nn.Linear(proj, proj, bias=False)

        # Near-zero init so dynamic path starts invisible
        nn.init.normal_(self.dynamic_up.weight,   std=0.01)
        nn.init.normal_(self.dynamic_gate.weight, std=0.01)
        nn.init.zeros_(self.dynamic_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        static_out  = self.static_mlp(x)
        gate        = F.silu(self.dynamic_gate(x))
        dynamic_out = self.dynamic_down(gate * self.dynamic_up(x))
        # Normalize only the dynamic delta — preserve the residual stream
        return static_out + self.norm(dynamic_out)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        keys      = self.key_proj(x)
        vals      = self.val_proj(x).detach()
        pred_vals = self.val_pred(keys)
        return F.mse_loss(pred_vals, vals)

    def dynamic_state(self) -> Dict:
        keys = [
            "dynamic_up.weight", "dynamic_gate.weight", "dynamic_down.weight",
            "norm.weight", "norm.bias",
            "key_proj.weight", "val_proj.weight", "val_pred.weight",
        ]
        sd = self.state_dict()
        return {k: sd[k].clone().cpu() for k in keys if k in sd}

    def load_dynamic_state(self, state: Dict):
        current = self.state_dict()
        for k, v in state.items():
            if k in current:
                current[k] = v.to(DEVICE)
        self.load_state_dict(current)


# ─────────────────────────────────────────────────────────────
# PONDER-TTT GATE
# ─────────────────────────────────────────────────────────────

class ReconstructionGate:
    def __init__(self, alpha: float = EMA_ALPHA,
                 threshold: float = EMA_THRESHOLD):
        self.ema       = None
        self.alpha     = alpha
        self.threshold = threshold
        self.n_fired   = 0
        self.n_skipped = 0

    def should_update(self, loss_val: float) -> bool:
        if self.ema is None:
            self.ema = loss_val
            self.n_fired += 1
            return True
        fire = loss_val > (self.threshold * self.ema)
        self.ema = self.alpha * loss_val + (1.0 - self.alpha) * self.ema
        if fire:
            self.n_fired += 1
        else:
            self.n_skipped += 1
        return fire

    @property
    def efficiency(self) -> float:
        total = self.n_fired + self.n_skipped
        return self.n_skipped / total if total > 0 else 0.0

    def state(self) -> Dict:
        return {
            "ema":       self.ema,
            "n_fired":   self.n_fired,
            "n_skipped": self.n_skipped,
        }

    def load_state(self, state: Dict):
        self.ema       = state.get("ema")
        self.n_fired   = state.get("n_fired", 0)
        self.n_skipped = state.get("n_skipped", 0)


# ─────────────────────────────────────────────────────────────
# TTT MODEL WRAPPER
# Handles both native TTT models and retrofit standard models
# ─────────────────────────────────────────────────────────────

class TTTModelWrapper:

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.gate     = ReconstructionGate()

        print(f"\n  Loading model: {model_id}")
        print(f"  Device: {DEVICE} | Dtype: {DTYPE}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
        )
        self.model.eval()

        # Detect if this is a native TTT model
        self.is_native_ttt = self._detect_native_ttt()

        if self.is_native_ttt:
            print("  ✓ Native TTT architecture detected — using inner-loop mode")
            self._setup_native_ttt()
        else:
            print("  ✓ Standard transformer — installing DualMLP TTT retrofit")
            self._setup_retrofit_ttt()

        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} params "
              f"({100*trainable/total:.2f}%)")
        print(f"  VRAM: {self.vram_usage()}\n")

    def _detect_native_ttt(self) -> bool:
        """Check if the model has native TTT layers (RetentionLabs models)."""
        model_str  = str(type(self.model)).lower()
        config_str = str(vars(self.model.config)).lower()
        ttt_signals = ["ttt", "test_time", "testtime", "retentionlabs"]
        return any(s in model_str or s in config_str for s in ttt_signals)

    # ── Native TTT setup ─────────────────────────────────────────────

    def _setup_native_ttt(self):
        """
        For RetentionLabs TTT models — the TTT layers already exist.
        We just need to identify which parameters should receive inner-loop
        updates and set up the optimizer.
        """
        for p in self.model.parameters():
            p.requires_grad = False

        ttt_params = []
        for name, param in self.model.named_parameters():
            if any(k in name.lower() for k in ["ttt", "fast_weight", "w1", "w2",
                                                 "inner", "delta"]):
                param.requires_grad = True
                ttt_params.append(param)

        if not ttt_params:
            print("  Warning: No explicit TTT params found, using layer split.")
            self._unfreeze_last_quarter()
            ttt_params = [p for p in self.model.parameters()
                         if p.requires_grad]

        lr = INNER_LR_MLP
        if "linear" in self.model_id.lower():
            lr = INNER_LR_LINEAR

        self.optimizer = torch.optim.Adam(ttt_params, lr=lr)
        self.ttt_heads = {}
        self.ttt_block_indices = []

    def _unfreeze_last_quarter(self):
        blocks = self._get_transformer_blocks()
        cutoff = math.ceil(len(blocks) * FREEZE_FRACTION)
        for i, block in enumerate(blocks):
            if i >= cutoff:
                for p in block.parameters():
                    p.requires_grad = True

    # ── Retrofit TTT setup (standard transformer) ─────────────────────

    def _setup_retrofit_ttt(self):
        """
        For granite, Qwen, etc. — surgically replace MLP blocks
        in the last 25% of layers with DualMLPTTTHead modules.
        """
        for p in self.model.parameters():
            p.requires_grad = False

        blocks  = self._get_transformer_blocks()
        n       = len(blocks)
        cutoff  = math.ceil(n * FREEZE_FRACTION)
        self.ttt_block_indices = list(range(cutoff, n))

        hidden_size = self.model.config.hidden_size
        self.ttt_heads: Dict[int, DualMLPTTTHead] = {}

        for idx in self.ttt_block_indices:
            block        = blocks[idx]
            original_mlp = self._get_mlp(block)
            head         = DualMLPTTTHead(
                original_mlp, hidden_size
            ).to(DEVICE).to(DTYPE)
            self._set_mlp(block, head)
            self.ttt_heads[idx] = head

        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=INNER_LR_MLP,
        )

        print(f"  TTT blocks: {self.ttt_block_indices[0]}–"
              f"{self.ttt_block_indices[-1]} "
              f"({len(self.ttt_block_indices)} of {n} blocks)")

    # ── Block introspection helpers ──────────────────────────────────

    def _get_transformer_blocks(self) -> list:
        m = self.model
        for path in ["model.layers", "transformer.h", "model.h",
                     "layers", "h", "blocks"]:
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError(f"Cannot find transformer blocks in {type(m)}")

    def _get_mlp(self, block) -> nn.Module:
        for attr in ["mlp", "feed_forward", "ff", "ffn", "fc"]:
            if hasattr(block, attr):
                return getattr(block, attr)
        raise RuntimeError(f"Cannot find MLP in block {type(block)}")

    def _set_mlp(self, block, mlp: nn.Module):
        for attr in ["mlp", "feed_forward", "ff", "ffn", "fc"]:
            if hasattr(block, attr):
                setattr(block, attr, mlp)
                return
        raise RuntimeError(f"Cannot set MLP in block {type(block)}")

    # ── Inner-loop adaptation ────────────────────────────────────────

    def adapt(self, text: str) -> Dict:
        """
        Run TTT inner-loop adaptation on a chunk of text.
        Should be called AFTER generation, not before.
        Returns stats dict.
        """
        if not text.strip():
            return {"updated": False, "loss": 0.0, "reason": "empty"}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=CHUNK_TOKENS,
        ).to(DEVICE)

        if inputs["input_ids"].shape[1] < 2:
            return {"updated": False, "loss": 0.0, "reason": "too_short"}

        # ── Probe: compute loss without updating ─────────────────────
        with torch.no_grad():
            if self.is_native_ttt:
                labels     = inputs["input_ids"].clone()
                out        = self.model(**inputs, labels=labels)
                probe_loss = out.loss.item()
            else:
                out = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs     = out.hidden_states
                losses = [
                    self.ttt_heads[idx].reconstruction_loss(hs[idx].to(DTYPE))
                    for idx in self.ttt_block_indices
                ]
                probe_loss = sum(l.item() for l in losses) / len(losses)

        # ── PonderTTT gate ────────────────────────────────────────────
        if not self.gate.should_update(probe_loss):
            return {
                "updated": False,
                "loss":    probe_loss,
                "ema":     self.gate.ema,
                "reason":  "gated",
            }

        # ── Inner-loop gradient step ──────────────────────────────────
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=DEVICE)

        seq_len  = inputs["input_ids"].shape[1]
        n_chunks = 0

        for start in range(0, seq_len, MINI_BATCH):
            end   = min(start + MINI_BATCH, seq_len)
            chunk = {k: v[:, start:end] for k, v in inputs.items()}

            if self.is_native_ttt:
                labels = chunk["input_ids"].clone()
                out    = self.model(**chunk, labels=labels)
                loss   = out.loss
            else:
                out = self.model(
                    **chunk,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs   = out.hidden_states
                loss = sum(
                    self.ttt_heads[idx].reconstruction_loss(hs[idx].to(DTYPE))
                    for idx in self.ttt_block_indices
                )

            total_loss = total_loss + loss
            n_chunks  += 1

        total_loss.backward()

        all_grads = [
            p for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        nn.utils.clip_grad_norm_(all_grads, MAX_GRAD_NORM)
        self.optimizer.step()

        # Clamp dynamic weights so they can never dominate the static path.
        # Without this, the reconstruction loss collapses to near-zero in one
        # step, the dynamic weights blow up, and generation corrupts on the
        # next turn.
        if not self.is_native_ttt:
            with torch.no_grad():
                for head in self.ttt_heads.values():
                    for param in [head.dynamic_up.weight,
                                  head.dynamic_gate.weight,
                                  head.dynamic_down.weight]:
                        param.clamp_(-0.1, 0.1)

        self.model.eval()

        return {
            "updated": True,
            "loss":    total_loss.item() / max(n_chunks, 1),
            "ema":     self.gate.ema,
            "reason":  "updated",
        }

    # ── Generation ───────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(self, messages) -> str:
        """
        Generate a response from a messages list or plain string.

        For Qwen3.5: the chat template automatically injects a closed
        <think></think> block when enable_thinking is not True, suppressing
        the reasoning scratchpad and producing clean responses. We patch
        the prompt string directly so this works on any transformers version.
        """
        if isinstance(messages, list):
            # ── Render template to string (no tokenization yet) ───────
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # ── Qwen3.5 thinking-mode patch ───────────────────────────
            # The template ends with one of:
            #   a) '<|im_start|>assistant\n<think>\n'        ← open (bad)
            #   b) '<|im_start|>assistant\n<think>\n\n</think>\n\n'  ← closed (good)
            #   c) '<|im_start|>assistant\n'                 ← no think block
            # We ensure it always ends in the closed form (b).
            OPEN_THINK   = "<|im_start|>assistant\n<think>\n"
            CLOSED_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            BARE_ASST    = "<|im_start|>assistant\n"

            if prompt_text.endswith(CLOSED_THINK):
                pass  # already correct
            elif prompt_text.endswith(OPEN_THINK):
                prompt_text = prompt_text[:-len(OPEN_THINK)] + CLOSED_THINK
            elif prompt_text.endswith(BARE_ASST):
                prompt_text = prompt_text[:-len(BARE_ASST)] + CLOSED_THINK

            # ── Tokenize the patched string ───────────────────────────
            enc = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
                add_special_tokens=False,  # template already added them
            )
            prompt_ids = enc.input_ids
        else:
            enc = self.tokenizer(
                messages,
                return_tensors="pt",
                truncation=True,
                max_length=3072,
            )
            prompt_ids = enc.input_ids

        prompt_ids     = prompt_ids.to(DEVICE)
        input_len      = prompt_ids.shape[1]
        attention_mask = torch.ones_like(prompt_ids)

        # ── Stop tokens ───────────────────────────────────────────────
        stop_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        for token in ["<|im_end|>", "<|endoftext|>", "</s>", "</think>"]:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if tid and tid != self.tokenizer.unk_token_id:
                stop_ids.append(tid)
        stop_ids = list(set(stop_ids)) or None

        # ── Generate ──────────────────────────────────────────────────
        output_ids = self.model.generate(
            prompt_ids,
            attention_mask     = attention_mask,
            max_new_tokens     = MAX_NEW_TOKENS,
            do_sample          = True,
            temperature        = TEMPERATURE,
            top_p              = TOP_P,
            eos_token_id       = stop_ids,
            pad_token_id       = self.tokenizer.pad_token_id
                                 or self.tokenizer.eos_token_id,
            use_cache          = True,
            repetition_penalty = REPETITION_PENALTY,
        )

        # ── Decode only new tokens ────────────────────────────────────
        new_ids  = output_ids[0, input_len:]
        response = self.tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip()

        print(response)
        return response

    # ── Weight delta serialization ────────────────────────────────────

    def serialize_deltas(self) -> bytes:
        """Serialize all trainable weight deltas to bytes for DB storage."""
        if self.is_native_ttt:
            state = {
                name: param.detach().cpu()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
        else:
            state = {
                idx: head.dynamic_state()
                for idx, head in self.ttt_heads.items()
            }
        buf = io.BytesIO()
        torch.save(state, buf)
        return buf.getvalue()

    def load_deltas(self, blob: bytes):
        """Restore serialized weight deltas."""
        buf   = io.BytesIO(blob)
        state = torch.load(buf, map_location=DEVICE, weights_only=False)

        if self.is_native_ttt:
            current = dict(self.model.named_parameters())
            for name, tensor in state.items():
                if name in current and current[name].requires_grad:
                    current[name].data.copy_(tensor.to(DEVICE))
        else:
            for idx, head_state in state.items():
                if idx in self.ttt_heads:
                    self.ttt_heads[idx].load_dynamic_state(head_state)

    def vram_usage(self) -> str:
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{used:.2f}/{total:.2f} GB"
        return "CPU"