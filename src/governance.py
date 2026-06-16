"""
governance.py — V3 (curated-source NLP/LLM layer added).
"""

import json
import datetime
import logging
from dataclasses import dataclass, field, asdict

from config import (
    TICKERS,
    TRAIN_START, TRAIN_END,
    VAL_START,   VAL_END,
    TEST_START,  TEST_END,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_RANDOM_STATE,
    RF_CLASS_WEIGHT, SIGNAL_THRESHOLD, TRANSACTION_COST_BPS,
    NLP_SOURCE_DIR, NLP_MODE, NLP_SIGNAL_DECAY_DAYS, NLP_WEIGHT, V2_PROXY_WEIGHT,
)

logger = logging.getLogger(__name__)


@dataclass
class GovernanceRecord:
    trained_at:           str   = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    tickers:              list  = field(default_factory=lambda: list(TICKERS))
    train_start:          str   = TRAIN_START
    train_end:            str   = TRAIN_END
    val_start:            str   = VAL_START
    val_end:              str   = VAL_END
    test_start:           str   = TEST_START
    test_end:             str   = TEST_END
    rf_n_estimators:      int   = RF_N_ESTIMATORS
    rf_max_depth:         int   = RF_MAX_DEPTH
    rf_min_samples_leaf:  int   = RF_MIN_SAMPLES_LEAF
    rf_random_state:      int   = RF_RANDOM_STATE
    rf_class_weight:      str   = RF_CLASS_WEIGHT
    signal_threshold:     float = SIGNAL_THRESHOLD
    transaction_cost_bps: int   = TRANSACTION_COST_BPS
    nlp_source_dir:       str   = NLP_SOURCE_DIR
    nlp_mode:             str   = NLP_MODE
    nlp_signal_decay_days:int   = NLP_SIGNAL_DECAY_DAYS
    nlp_weight:           float = NLP_WEIGHT
    v2_proxy_weight:      float = V2_PROXY_WEIGHT
    ai_signal_version:    str   = "V3-nlp-enriched"
    ai_makes_decisions:   bool  = False
    is_live_trading:      bool  = False
    is_investment_advice: bool  = False

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "## Model Governance Report",
            "",
            f"**Generated at (UTC):** {self.trained_at}",
            "",
            "---",
            "",
            "### Research Objective",
            "",
            "> The goal of this prototype is **not** to maximise backtest return.  ",
            "> The goal is to test whether contextual information — first price-derived,  ",
            "> now text-derived — improves robustness across market regimes.  ",
            "> Strong training performance is insufficient.  ",
            "> Validation and test performance must be interpreted independently.  ",
            "> NLP signals may be regime-dependent and data-limited.",
            "",
            "---",
            "",
            "### Version History",
            "",
            "| Version | Status | Description |",
            "|---------|--------|-------------|",
            "| V1 | ✅ Done | Neutral mock — all signals = 0.0. |",
            "| V2 | ✅ Done | Market-derived price-proxy signals. |",
            "| V2.1 | ✅ Done | Research diagnostics — ablation, rolling, permutation importance, regime. |",
            "| V3 | ✅ Current | Curated-source NLP/LLM contextual signal layer. |",
            "| V4 | 🔲 Planned | Broader source coverage, streaming ingestion, live NLP confidence tracking. |",
            "",
            "> **The Random Forest is the sole decision-making layer in all versions.**  ",
            "> NLP/LLM output is validated against a schema and used only as numerical features.",
            "",
            "---",
            "",
            "### V3 NLP Layer Design",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Source directory | `{self.nlp_source_dir}` |",
            f"| Extraction mode  | `{self.nlp_mode}` |",
            f"| Signal decay     | {self.nlp_signal_decay_days} days |",
            f"| NLP blend weight | {self.nlp_weight} |",
            f"| V2 proxy weight  | {self.v2_proxy_weight} |",
            "",
            "### Temporal Split",
            "| Period | Start | End |",
            "|--------|-------|-----|",
            f"| Training   | {self.train_start} | {self.train_end} |",
            f"| Validation | {self.val_start}   | {self.val_end}   |",
            f"| Test       | {self.test_start}  | {self.test_end}  |",
            "",
            "### Model Configuration",
            f"- Estimators : {self.rf_n_estimators}",
            f"- Max depth  : {self.rf_max_depth}",
            f"- Min leaf   : {self.rf_min_samples_leaf}",
            f"- Class weight: {self.rf_class_weight}",
            f"- Random seed: {self.rf_random_state}",
            f"- Signal threshold: {self.signal_threshold}",
            "",
            "### Transaction Costs",
            f"- {self.transaction_cost_bps} bps per unit of daily turnover (strategy only).",
            "",
            "### Compliance Statements",
            f"- Signal version         : **{self.ai_signal_version}**",
            f"- LLM / Signal decides   : **{self.ai_makes_decisions}**",
            f"- Live trading system    : **{self.is_live_trading}**",
            f"- Investment advice      : **{self.is_investment_advice}**",
            "",
            "### Governance Rules",
            "1.  Training data bounded to [TRAIN_START, TRAIN_END].",
            "2.  No hyperparameter changed after the validation period was observed.",
            "3.  Any structural change requires full retraining.",
            "4.  All price-proxy signals use trailing-only rolling windows.",
            "5.  Text sources are local, curated, and auditable — no web browsing.",
            "6.  LLM reads ONLY the provided document text; no outside knowledge.",
            "7.  LLM does NOT make trading decisions.",
            "8.  LLM output is validated against nlp_schema.py before use.",
            "9.  Invalid LLM output falls back safely to heuristic extraction.",
            "10. A text source dated D can only affect market dates >= D (no future leakage).",
            "11. NLP signals are features only; they are not position signals.",
            "12. Transaction costs applied to every strategy evaluation.",
            "13. Diagnostic analyses (ablation, regime, permutation) never modify model parameters.",
            "14. This prototype is for research and educational use only.",
        ]
        return "\n".join(lines)


def create_record() -> GovernanceRecord:
    record = GovernanceRecord()
    logger.info("Governance record created:\n%s", record.to_json())
    return record


DISCLAIMER = """
---
⚠️ **DISCLAIMER — RESEARCH PROTOTYPE ONLY**

This application is a **research and educational prototype**.  
It is **not** a live trading system.  
It is **not** investment advice.  
It is **not** a regulated investment product or fund.  
Past simulated performance does not guarantee future results.  
All results shown are hypothetical and based on historical data.  
No real money is at risk. Do not make investment decisions based on this tool.

The V3 NLP layer reads only curated local source documents.  
It does not browse randomly. It does not make trading decisions.  
The Random Forest remains the sole decision-making layer.

---
"""
