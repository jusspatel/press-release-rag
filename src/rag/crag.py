import re
from enum import Enum
from typing import Optional
import torch
from pydantic import BaseModel, Field


class CRAGStatus(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class DocGradeDecision(BaseModel):
    status: CRAGStatus
    missing_facts: Optional[str] = Field(
        default=None,
        description="Factual information missing from the retrieved text, if ambiguous or incorrect.",
    )
    fallback_search_query: Optional[str] = Field(
        default=None,
        description="Optimized keyword query to find missing details via web search.",
    )
    reasoning: str


class StripEvaluation(BaseModel):
    is_relevant: bool = Field(
        description="True if this specific strip contains verified factual evidence answering the query."
    )


class CRAGEngine:
    def __init__(self, tokenizer, model, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def decompose(self, text: str) -> list[str]:
        """Splits retrieved passage text into fine-grained knowledge strips (1-2 sentences)."""
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 20
        ]
        strips = []
        for i in range(0, len(sentences), 2):
            strip = " ".join(sentences[i : i + 2]).strip()
            if len(strip) > 30:
                strips.append(strip)
        return strips

    def evaluate_documents(self, query: str, chunks: list[dict]) -> DocGradeDecision:
        """First-pass evaluation: Grade collective retrieved passages as Correct, Ambiguous, or Incorrect."""
        combined_text = "\n\n".join(
            f"Document {i+1} [{c.get('title', '')}]:\n{c.get('text', '')}"
            for i, c in enumerate(chunks)
        )

        prompt = (
            f"<|im_start|>system\n"
            f"You are a CRAG retrieval evaluator assessing official 2026 Indian Government PIB excerpts against a user query.\n"
            f"Determine if the documents are:\n"
            f"- 'correct': Direct, sufficient facts are present.\n"
            f"- 'ambiguous': Partially relevant, but missing key statistics, figures, or specific answers.\n"
            f"- 'incorrect': Completely irrelevant or fails to answer the inquiry.\n\n"
            f"Respond ONLY with valid JSON matching:\n"
            f'{{"status": "correct"|"ambiguous"|"incorrect", "missing_facts": "string or null", "fallback_search_query": "string or null", "reasoning": "string"}}<|im_end|>\n'
            f"<|im_start|>user\n"
            f"USER QUERY: {query}\n\n"
            f"CONTEXT:\n{combined_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=200, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        raw_text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        candidate = match.group(1).strip() if match else raw_text
        return DocGradeDecision.model_validate_json(candidate)

    def compact_context(
        self,
        query: str,
        passages: list[str] | str,
        max_tokens: int = 400,
    ) -> str:
        """
        Compacts retrieved passages into a dense, non-redundant factual context
        using the local small LLM (Qwen 2.5 1.5B) in a single inference call.
        """
        if isinstance(passages, list):
            combined_text = "\n\n".join(p.strip() for p in passages if p and p.strip())
        else:
            combined_text = passages.strip() if passages else ""

        if not combined_text:
            return ""

        prompt = (
            f"<|im_start|>system\n"
            f"You are a factual knowledge compaction engine for an Indian governance intelligence system.\n"
            f"Your task is to compress and compact the provided reference excerpts into a dense, accurate factual summary directly relevant to the user query.\n"
            f"Rules:\n"
            f"1. Extract ONLY specific facts, figures, crop names, rates/prices, dates, percentages, organizations, and cabinet decisions that help answer the query.\n"
            f"2. Completely remove bureaucratic greetings, administrative boilerplate, speaker titles, website navigation, and redundant statements.\n"
            f"3. Strictly preserve exact numbers, proper nouns, and policy details. Do not extrapolate, infer, or hallucinate.\n"
            f"4. Format the output as concise, high-density factual bullet points.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"USER QUERY: {query}\n\n"
            f"SOURCE EXCERPTS:\n{combined_text[:4000]}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        compacted = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        return compacted if compacted else combined_text[:1200]

    def filter_and_recompose(self, query: str, chunks: list[dict]) -> str:
        """Extracts chunk texts and runs single-pass knowledge compaction."""
        raw_texts = [
            f"[{c.get('title', 'PIB Record')}]: {c.get('text', '')}"
            for c in chunks
            if c.get("text")
        ]
        if not raw_texts:
            return ""
        return self.compact_context(query, raw_texts)

    def filter_and_recompose_strips(self, query: str, chunks: list[dict]) -> str:
        """Legacy strip evaluator: Decomposes chunks into atomic strips, evaluates each individually, and discards noise."""
        all_strips = []
        for chunk in chunks:
            all_strips.extend(self.decompose(chunk.get("text", "")))

        verified_strips = []
        for strip in all_strips:
            prompt = (
                f"<|im_start|>system\n"
                f"Evaluate if this atomic strip contains relevant factual information answering or directly relating to the query.\n"
                f'Output ONLY JSON: {{"is_relevant": true|false}}<|im_end|>\n'
                f"<|im_start|>user\n"
                f"QUERY: {query}\n"
                f"STRIP: {strip}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=40, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
                )
            res_str = self.tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            match = re.search(r"(\{.*\})", res_str, re.DOTALL)
            try:
                eval_obj = StripEvaluation.model_validate_json(
                    match.group(1).strip() if match else res_str
                )
                if eval_obj.is_relevant:
                    verified_strips.append(strip)
            except Exception:
                # If parsing fails on an atomic strip, retain it as a safety fallback
                verified_strips.append(strip)

        if not verified_strips and chunks:
            return chunks[0].get("text", "")

        return "\n\n".join(verified_strips)