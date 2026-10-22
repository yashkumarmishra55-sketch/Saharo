"""
OmniCognitive V5 — Adaptive Information-Theoretic Logits Processor
===================================================================
Author: Yash Kumar Mishra
Location: Nagpur, Maharashtra, India (India's Zero Mile)

This module provides a high-performance, batch-aware LogitsProcessor that dynamically
adjusts sampling behavior based on Shannon entropy, varentropy, and generation history.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import LogitsProcessor

logger = logging.getLogger("OmniCognitiveV5")


@dataclass(slots=True)
class UltraTelemetry:
    """Lightweight per-step diagnostics collector."""
    entropies: List[float] = field(default_factory=list)
    varentropies: List[float] = field(default_factory=list)
    effective_temps: List[float] = field(default_factory=list)
    surviving_tokens: List[float] = field(default_factory=list)
    chaos_streaks: List[int] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        if not self.entropies:
            return {}
        n = len(self.entropies)
        sorted_tokens = sorted(self.surviving_tokens)
        
        def pct(p: float) -> float:
            return float(sorted_tokens[min(int(n * p), n - 1)]) if n else 0.0

        mean_h = sum(self.entropies) / n
        mean_v = sum(self.varentropies) / n
        
        return {
            "total_steps": n,
            "mean_entropy_nats": round(mean_h, 4),
            "mean_varentropy": round(mean_v, 4),
            "tokens_p50": pct(0.50),
            "tokens_p95": pct(0.95),
            "tokens_p99": pct(0.99),
        }

    def reset(self) -> None:
        self.entropies.clear()
        self.varentropies.clear()
        self.effective_temps.clear()
        self.surviving_tokens.clear()
        self.chaos_streaks.clear()


class OmniCognitiveV5(LogitsProcessor):
    """Adaptive logits processor with 4 novel mechanisms."""

    def __init__(
        self,
        prompt_len: int,
        base_threshold: float = 1.2,
        alpha: float = 0.08,
        varentropy_weight: float = 0.15,
        k_min: int = 2,
        k_max: int = 32,
        min_p: float = 0.08,
        temp_min: float = 0.35,
        cooling_rate: float = 0.30,
        decay_lambda: float = 0.45,
        repetition_penalty: float = 1.15,
        frequency_penalty: float = 0.15,
        presence_penalty: float = 0.20,
        dry_n: int = 4,
        chaos_streak_limit: int = 3,
        record_telemetry: bool = True,
    ) -> None:
        if prompt_len < 0:
            raise ValueError("prompt_len must be non-negative")
        if k_min < 1 or k_max < k_min:
            raise ValueError("Invalid k_min / k_max combination")
        if not (0.0 < min_p <= 1.0):
            raise ValueError("min_p must be in (0, 1]")

        self.prompt_len = prompt_len
        self.base_threshold = float(base_threshold)
        self.alpha = float(alpha)
        self.gamma = float(varentropy_weight)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.min_p = float(min_p)
        self.temp_min = float(temp_min)
        self.beta = float(cooling_rate)
        self.decay_lambda = float(decay_lambda)
        self.rep_penalty = float(repetition_penalty)
        self.freq_penalty = float(frequency_penalty)
        self.pres_penalty = float(presence_penalty)
        self.dry_n = int(dry_n)
        self.chaos_limit = int(chaos_streak_limit)

        self.record_telemetry = bool(record_telemetry)
        self.telemetry = UltraTelemetry()
        self._chaos_streaks: Optional[torch.Tensor] = None
        self._freq_buf: Optional[torch.Tensor] = None
        self._freq_buf_key: Optional[tuple] = None

    def reset_state(self) -> None:
        self._chaos_streaks = None
        self._freq_buf = None
        self._freq_buf_key = None
        self.telemetry.reset()

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device
        orig_dtype = scores.dtype
        B, V = scores.shape

        if self.prompt_len > input_ids.shape[-1]:
            raise ValueError(f"prompt_len={self.prompt_len} > input length {input_ids.shape[-1]}")

        if self._chaos_streaks is None or self._chaos_streaks.shape[0] != B:
            self._chaos_streaks = torch.zeros(B, dtype=torch.long, device=device)

        history = input_ids[:, self.prompt_len:]
        if history.shape[1] == 0:
            self._chaos_streaks.zero_()

        s = scores.float()

        # 1. Penalties
        if history.shape[1] > 0:
            key = (B, V, str(device))
            if self._freq_buf is None or self._freq_buf_key != key:
                self._freq_buf = torch.zeros((B, V), dtype=torch.float32, device=device)
                self._freq_buf_key = key
            else:
                self._freq_buf.zero_()
            
            self._freq_buf.scatter_add_(1, history, torch.ones_like(history, dtype=torch.float32))
            freq = self._freq_buf
            appeared = freq > 0.0

            s = torch.where(appeared, torch.where(s > 0, s / self.rep_penalty, s * self.rep_penalty), s)
            s = s - freq * self.freq_penalty - appeared.to(torch.float32) * self.pres_penalty

        # 2. Entropy & Varentropy
        log_probs = torch.log_softmax(s, dim=-1)
        if not torch.isfinite(log_probs).all():
            return scores

        probs = log_probs.exp()
        valid = probs > 0.0
        surprisal = torch.where(valid, -log_probs, torch.zeros_like(log_probs))
        entropy = (probs * surprisal).sum(dim=-1)
        centered = torch.where(valid, surprisal - entropy.unsqueeze(-1), torch.zeros_like(surprisal))
        varentropy = (probs * centered * centered).sum(dim=-1)

        # 3. Dynamic Threshold
        gen_steps = max(1, input_ids.shape[-1] - self.prompt_len)
        dynamic_threshold = self.base_threshold + self.alpha * math.log1p(gen_steps)
        excess = torch.clamp(entropy - dynamic_threshold, min=0.0)
        instability = excess * (1.0 + self.gamma * varentropy)

        # 4. Chaos Modifier
        chaos_detected = excess > 0.5
        self._chaos_streaks = torch.where(chaos_detected, self._chaos_streaks + 1, torch.zeros_like(self._chaos_streaks))
        chaos_mod = torch.where(self._chaos_streaks >= self.chaos_limit, torch.full_like(instability, float(self.dry_n)), torch.ones_like(instability))

        # 5. Adaptive Temp
        effective_temp = torch.clamp(1.0 / (1.0 + self.beta * instability * chaos_mod), min=self.temp_min, max=1.0)
        scaled = s / effective_temp.unsqueeze(-1)

        # 6. Top-K + Min-P
        scaled_lp = torch.log_softmax(scaled, dim=-1)
        max_lp = scaled_lp.amax(dim=-1, keepdim=True)
        min_p_mask = scaled_lp >= (max_lp + math.log(self.min_p))

        uncertain = excess > 0.0
        k_adapt = torch.clamp(torch.round(self.k_min + (self.k_max - self.k_min) * torch.exp(-self.decay_lambda * instability)).long(), min=self.k_min, max=min(self.k_max, V))
        row_k = torch.where(uncertain, k_adapt, torch.ones_like(k_adapt))
        max_k = min(int(row_k.max().item()), V)

        _, top_idx = torch.topk(scaled, k=max_k, dim=-1)
        ranks = torch.arange(max_k, device=device).unsqueeze(0)
        topk_mask = torch.zeros_like(scaled, dtype=torch.bool)
        topk_mask.scatter_(-1, top_idx, ranks < row_k.unsqueeze(1))

        keep = torch.where(uncertain.unsqueeze(-1), topk_mask & min_p_mask, min_p_mask)
        keep.scatter_(-1, torch.argmax(scaled, dim=-1, keepdim=True), True)
        
        output = scaled.masked_fill(~keep, float("-inf"))

        if self.record_telemetry:
            self.telemetry.entropies.append(float(entropy.mean().item()))
            self.telemetry.varentropies.append(float(varentropy.mean().item()))
            self.telemetry.effective_temps.append(float(effective_temp.mean().item()))
            self.telemetry.surviving_tokens.append(float(keep.sum(dim=-1).float().mean().item()))
            self.telemetry.chaos_streaks.append(int(self._chaos_streaks.max().item()))

        return output.to(orig_dtype)
