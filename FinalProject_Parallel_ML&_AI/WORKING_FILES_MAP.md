# Working Files Mapping - 6.7% Accuracy Implementation

**Date**: November 20, 2025
**Status**: ✅ Working (Achieved 6.7% success at episode 15)
**Purpose**: Document exactly which files were used to achieve 6.7% accuracy

---

## Directory Structure Overview

```
HPC_ParallerML/
├── FinalProject_Parallel_ML&_AI/          # Main project directory
│   ├── modal_train_apo_only.py           ✅ WORKING - Main training script (achieved 6.7%)
│   ├── ragen/                             # RAGEN implementation
│   │   └── environments/
│   │       └── simple_webshop_mock.py    ✅ WORKING - SimpleMockWebShop (5 products)
│   ├── tinyzero/                          # RL trainers and models
│   │   ├── multiturn_apo_trainer.py      ✅ WORKING - MultiTurnAPOTrainer
│   │   └── models.py                     ✅ WORKING - PolicyModel wrapper
│   └── [many other files - OLD/UNUSED]
│
└── WebShop/                               # Real WebShop data repository
    └── data/
        ├── items_human_ins.json          ✅ REAL DATA - 5.1MB, real WebShop products
        ├── items_ins_v2_1000.json        📦 Alternative dataset (147KB)
        └── items_shuffle_1000.json       📦 Alternative dataset (4.5MB)
```

---

## ✅ WORKING FILES (Used to achieve 6.7%)

### 1. Main Training Script

**File**: `modal_train_apo_only.py`
**Path**: `/Users/nikhilpandey/Parallel_ML/parallel_ml_repo/HPC_ParallerML/FinalProject_Parallel_ML&_AI/modal_train_apo_only.py`
**Status**: ✅ **WORKING** - Achieved 6.7% success
**Purpose**: Pure APO training (no BC) on SimpleMockWebShop

**Key imports**:
```python
from ragen.environments.simple_webshop_mock import SimpleMockWebShop
from tinyzero.multiturn_apo_trainer import MultiTurnAPOTrainer
from tinyzero.models import PolicyModel
```

**Configuration**:
- Model: `Qwen/Qwen2.5-3B-Instruct`
- GPU: H100 (80GB)
- Learning rate: `1e-5`
- Episodes: `50`
- Environment: `SimpleMockWebShop()` (5 products)

**Result**: 6.7% success at episode 15, degraded to 0% by episode 50

---

### 2. SimpleMockWebShop Environment

**File**: `simple_webshop_mock.py`
**Path**: `/Users/nikhilpandey/Parallel_ML/parallel_ml_repo/HPC_ParallerML/FinalProject_Parallel_ML&_AI/ragen/environments/simple_webshop_mock.py`
**Status**: ✅ **WORKING**
**Purpose**: Simplified 5-product WebShop environment

**Products**:
```python
products = {
    1: {"name": "red shoes", "keywords": ["red", "shoes"], "price": "$45"},
    2: {"name": "blue headphones", "keywords": ["blue", "headphones"], "price": "$89"},
    3: {"name": "black laptop", "keywords": ["black", "laptop"], "price": "$599"},
    4: {"name": "white shirt", "keywords": ["white", "shirt"], "price": "$25"},
    5: {"name": "green backpack", "keywords": ["green", "backpack"], "price": "$39"}
}
```

**Rewards**:
- Search (target in results): `+0.3`
- Click correct product: `+0.3`
- Buy correct product: `+1.0`
- Total possible reward: `1.6`

**Features**:
- Simple keyword matching
- Natural language actions (loose format)
- Immediate rewards
- 3-step tasks (search → click → buy)

---

### 3. MultiTurnAPOTrainer

**File**: `multiturn_apo_trainer.py`
**Path**: `/Users/nikhilpandey/Parallel_ML/parallel_ml_repo/HPC_ParallerML/FinalProject_Parallel_ML&_AI/tinyzero/multiturn_apo_trainer.py`
**Status**: ✅ **WORKING**
**Purpose**: APO trainer adapted for multi-turn sequential tasks

**Key methods**:
```python
class MultiTurnAPOTrainer:
    def rollout_episode(self, env, max_steps=10):
        """Generate full trajectory by interacting with environment"""

    def compute_advantages(self, rewards):
        """Compute advantages using GAE (simplified)"""

    def train_step(self, env, num_episodes=1):
        """Run episodes and train with APO"""
```

**Advantages over PPO**:
- Simpler loss function (no ratio clipping)
- No reference model needed during training
- More numerically stable
- No NaN crashes

**Configuration**:
- Beta: `0.1`
- Gamma: `1.0`
- GAE Lambda: `1.0`
- Temperature: `0.7`
- Top-p: `0.9`

---

### 4. PolicyModel Wrapper

**File**: `models.py`
**Path**: `/Users/nikhilpandey/Parallel_ML/parallel_ml_repo/HPC_ParallerML/FinalProject_Parallel_ML&_AI/tinyzero/models.py`
**Status**: ✅ **WORKING**
**Purpose**: Wrapper around HuggingFace models for RL training

**Key class**:
```python
class PolicyModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt):
        """Generate action from prompt"""

    def parameters(self):
        """Return model parameters for optimizer"""
```

---

## 📦 REAL WEBSHOP DATA

**Location**: `/Users/nikhilpandey/Parallel_ML/parallel_ml_repo/HPC_ParallerML/WebShop/data/`

### Main Dataset

**File**: `items_human_ins.json`
**Size**: 5.1 MB
**Status**: ✅ **REAL DATA**
**Purpose**: Real WebShop products with human instructions

**Structure**:
```json
[
    {
        "asin": "B09QKP7XQL",
        "instruction": "i am looking for wireless bluetooth headphones...",
        "name": "TOZO T6 True Wireless Earbuds",
        "attributes": {...},
        "price": "$49.99",
        ...
    },
    ...
]
```

**Usage**: Will be used for Stages 3-6 in curriculum learning

### Alternative Datasets

**File**: `items_ins_v2_1000.json`
**Size**: 147 KB
**Products**: ~1000 products

**File**: `items_shuffle_1000.json`
**Size**: 4.5 MB
**Products**: ~1000 products (shuffled)

---

## ❌ OLD/UNUSED FILES (Do NOT use for curriculum)

### Old Training Scripts (NOT working)

```
❌ modal_train_ragen.py              # Old RAGEN with PPO (failed)
❌ modal_train_ppo.py                 # PPO attempts (NaN crashes)
❌ modal_train_webshop_ppo.py         # WebShop PPO (failed)
❌ modal_train_bc_intensive.py        # BC experiments (0% success)
❌ modal_train_curriculum.py          # Old curriculum (didn't work)
❌ modal_train_curriculum_bc_ppo.py   # BC + PPO curriculum (failed)
❌ modal_train_curriculum_multiturn_ppo.py  # Multi-turn PPO (failed)
```

### Old Environments (NOT working)

```
❌ ragen/environments/webshop.py              # Old WebShop wrapper (complex)
❌ ragen/environments/simple_webshop.py       # Earlier simple version
❌ ragen/environments/medium_webshop.py       # Medium complexity (not tested)
❌ ragen/environments/medium_200_webshop.py   # 200 products (not tested)
❌ ragen/environments/simple_100_webshop.py   # 100 products (not tested)
```

### Old Trainers (NOT working)

```
❌ tinyzero/ppo_trainer.py            # PPO trainer (NaN issues)
❌ tinyzero/apo_trainer.py            # Single-turn APO (not adapted for WebShop)
❌ ragen/multi_turn_ppo_trainer.py    # Multi-turn PPO (crashed)
❌ ragen/agent_trainer.py             # Old agent trainer
❌ ragen/webshop_trainer.py           # Old WebShop trainer
```

### Old Data/Demo Files (NOT used)

```
❌ ragen/webshop_expert_demos.py      # Fake demos (caused mode collapse)
❌ ragen/expert_demos.py              # Old demo format
❌ ragen/real_expert_demos.py         # Real demos (distribution mismatch)
❌ ragen/medium_expert_demos.py       # Medium complexity demos
❌ ragen/medium_webshop_data.py       # Medium data loader
```

---

## 📋 FILE DEPENDENCY TREE

### Working Implementation (6.7% accuracy)

```
modal_train_apo_only.py
├── ragen/environments/simple_webshop_mock.py  ✅ (SimpleMockWebShop class)
├── tinyzero/multiturn_apo_trainer.py          ✅ (MultiTurnAPOTrainer class)
└── tinyzero/models.py                         ✅ (PolicyModel class)

Dependencies:
- torch==2.1.0
- transformers==4.37.0
- accelerate==0.27.0
- numpy==1.24.3
- gym==0.26.2
```

**No other files are needed!** The implementation is self-contained.

---

## 🎯 FOR CURRICULUM LEARNING

### Files We WILL Use (Building on working implementation)

**Stage 1** (Already working):
- ✅ `simple_webshop_mock.py` - Current 5-product environment
- ✅ `multiturn_apo_trainer.py` - Current APO trainer
- ✅ `models.py` - Current PolicyModel

**Stage 2** (To be created):
- 🆕 `ragen/environments/extended_webshop_mock.py` - 15 products, distractors
- ✅ `multiturn_apo_trainer.py` - Same trainer (reuse)
- ✅ `models.py` - Same model wrapper (reuse)

**Stage 3** (To be created):
- 🆕 `ragen/environments/complex_webshop_mock.py` - 50 products, codes, 20% real
- 🆕 `ragen/data/stage_3_real_products.json` - 10 easy real products from WebShop
- ✅ `multiturn_apo_trainer.py` - Same trainer (reuse)

**Stage 4** (To be created):
- 🆕 `ragen/environments/hybrid_webshop.py` - 100 products, 50% real
- 🆕 `ragen/data/stage_4_real_products.json` - 50 real products from WebShop
- ✅ `multiturn_apo_trainer.py` - Same trainer (reuse)

**Stage 5** (To be created):
- 🆕 `ragen/environments/mostly_real_webshop.py` - 500 products, 80% real
- 🆕 `ragen/data/stage_5_real_products.json` - 400 real products from WebShop
- ✅ `multiturn_apo_trainer.py` - Same trainer (reuse)

**Stage 6** (Use real WebShop):
- 🆕 `ragen/environments/real_webshop_wrapper.py` - Wrapper around WebShop data
- ✅ `WebShop/data/items_human_ins.json` - Full real dataset (5.1MB)
- ✅ `multiturn_apo_trainer.py` - Same trainer (reuse)

### Files We WILL Create (New for curriculum)

**Curriculum manager**:
- 🆕 `ragen/environments/curriculum_manager.py` - Manages stage transitions

**Data generation scripts**:
- 🆕 `scripts/generate_stage_2_products.py` - Generate 15 mock products
- 🆕 `scripts/generate_stage_3_products.py` - Generate 40 mock + sample 10 real
- 🆕 `scripts/generate_stage_4_products.py` - Generate 50 mock + sample 50 real
- 🆕 `scripts/generate_stage_5_products.py` - Generate 100 mock + sample 400 real
- 🆕 `scripts/sample_real_products.py` - Sample real products by difficulty

**Curriculum training script**:
- 🆕 `modal_train_curriculum_apo.py` - Main curriculum training script

---

## 🔑 KEY INSIGHTS

### What Works (Keep these!)

1. ✅ **SimpleMockWebShop** - 5 products, simple rewards, natural language
2. ✅ **MultiTurnAPOTrainer** - Stable, no NaN crashes
3. ✅ **PolicyModel wrapper** - Clean interface for HuggingFace models
4. ✅ **Pure APO (no BC)** - Simpler, more stable than BC + PPO

### What Doesn't Work (Ignore these!)

1. ❌ **PPO on WebShop** - NaN crashes after ~20 steps
2. ❌ **BC without observations** - Distribution mismatch, degrades performance
3. ❌ **Fake expert demos** - Mode collapse, 0% success
4. ❌ **Direct training on real WebShop** - Too complex, no learning signal

### What We Learned

1. **Start simple** - SimpleMockWebShop (5 products) works, real WebShop (1000s) doesn't
2. **APO > PPO** - More stable, simpler, no NaN issues
3. **Pure RL works** - 6.7% at episode 15 proves RL signal exists
4. **Catastrophic forgetting** - Main problem to solve (6.7% → 0%)

---

## 📝 USAGE GUIDE

### To Reproduce 6.7% Result

```bash
cd /Users/nikhilpandey/Parallel_ML/parallel_ml_repo/HPC_ParallerML/FinalProject_Parallel_ML&_AI/

# Run the working script
python modal_train_apo_only.py

# Expected result:
# - Episode 15: 6.7% success
# - Episode 50: 0% success (catastrophic forgetting)
```

### To Access Real WebShop Data

```python
import json

# Load real WebShop products
data_path = "/Users/nikhilpandey/Parallel_ML/parallel_ml_repo/HPC_ParallerML/WebShop/data/items_human_ins.json"

with open(data_path, 'r') as f:
    real_products = json.load(f)

print(f"Loaded {len(real_products)} real products")
# Output: Loaded ~12000 real products

# Example product structure
product = real_products[0]
print(product.keys())
# Output: dict_keys(['asin', 'instruction', 'name', 'attributes', 'price', ...])
```

### To Build Curriculum (Next Steps)

1. **Stage 2**: Create `extended_webshop_mock.py` with 15 products
2. **Stage 3**: Generate 40 mock + sample 10 real from `items_human_ins.json`
3. **Stage 4**: Generate 50 mock + sample 50 real
4. **Stage 5**: Generate 100 mock + sample 400 real
5. **Stage 6**: Use full `items_human_ins.json` dataset

---

## 🗂️ DIRECTORY CLEANUP RECOMMENDATIONS

### Files to Keep

```
FinalProject_Parallel_ML&_AI/
├── modal_train_apo_only.py                   ✅ KEEP
├── FINAL_PROJECT_ANALYSIS.md                 ✅ KEEP (analysis document)
├── CURRICULUM_LEARNING_PLAN.md               ✅ KEEP (new plan)
├── WORKING_FILES_MAP.md                      ✅ KEEP (this document)
├── ragen/
│   └── environments/
│       ├── simple_webshop_mock.py           ✅ KEEP
│       └── simple_maze.py                   ✅ KEEP (for testing)
├── tinyzero/
│   ├── multiturn_apo_trainer.py            ✅ KEEP
│   └── models.py                            ✅ KEEP
└── requirements.txt                         ✅ KEEP
```

### Files Can Archive (Old experiments)

```
FinalProject_Parallel_ML&_AI/
├── modal_train_*.py                          📦 ARCHIVE (except modal_train_apo_only.py)
├── ragen/
│   ├── train_ragen.py                       📦 ARCHIVE (old PPO)
│   ├── agent_trainer.py                     📦 ARCHIVE
│   ├── webshop_trainer.py                   📦 ARCHIVE
│   ├── multi_turn_ppo_trainer.py            📦 ARCHIVE
│   ├── webshop_expert_demos.py              📦 ARCHIVE (fake demos)
│   └── environments/
│       ├── webshop.py                       📦 ARCHIVE
│       ├── simple_webshop.py                📦 ARCHIVE
│       ├── medium_webshop.py                📦 ARCHIVE
│       └── simple_100_webshop.py            📦 ARCHIVE
└── tinyzero/
    ├── ppo_trainer.py                       📦 ARCHIVE
    └── apo_trainer.py                       📦 ARCHIVE (single-turn only)
```

---

## ✅ SUMMARY

**Working Implementation** (6.7% accuracy):
- **Environment**: `simple_webshop_mock.py` (5 products)
- **Trainer**: `multiturn_apo_trainer.py` (APO algorithm)
- **Model**: `models.py` (PolicyModel wrapper)
- **Script**: `modal_train_apo_only.py` (Pure APO, no BC)

**Real WebShop Data**:
- **Location**: `/WebShop/data/items_human_ins.json`
- **Size**: 5.1 MB (~12,000 products)
- **Usage**: Stages 3-6 in curriculum learning

**Next Steps**:
1. Build Stage 2 environment (15 products)
2. Sample real data for Stages 3-6
3. Implement curriculum manager
4. Run full curriculum training

**Key Principle**: Build on what works (simple_webshop_mock.py + multiturn_apo_trainer.py), ignore what failed (PPO, BC, complex WebShop wrappers).
