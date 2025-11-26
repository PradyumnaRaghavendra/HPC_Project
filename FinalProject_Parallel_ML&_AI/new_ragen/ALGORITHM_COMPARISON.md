# A*-PO Algorithm Comparison: TinyZero vs new_ragen

## Executive Summary

**CRITICAL FINDING**: TinyZero and new_ragen implement **fundamentally different algorithms**, despite both claiming to be "A*-PO":

- **TinyZero**: Weighted Supervised Learning (Teacher-Forcing) with advantage-based example weights
- **new_ragen**: Pure REINFORCE Policy Gradient with advantage weighting

This explains why TinyZero works and new_ragen doesn't. The algorithms are completely different.

---

## Detailed Comparison

### 1. Core Algorithm Approach

| Aspect | TinyZero (WORKING) | new_ragen (NOT WORKING) |
|--------|-------------------|-------------------------|
| **Algorithm Type** | Weighted Supervised Learning | REINFORCE Policy Gradient |
| **Training Mode** | Teacher-forcing on generated completions | Online trajectory rollouts |
| **Loss Function** | Weighted Cross-Entropy Loss | Policy Gradient Loss |
| **Loss Formula** | `loss = (CE_loss * weights).mean()` | `loss = -(log_prob * advantages).mean()` |

**TinyZero Algorithm**:
```python
# Step 1: Generate completions
generated_texts = policy.generate(prompts)

# Step 2: Compute rewards and advantages
rewards = [compute_reward_with_partial_credit(text, problem) for text in generated_texts]
advantages = rewards - V_star

# Step 3: Normalize advantages and convert to weights
adv_norm = (advantages - adv_mean) / adv_std
adv_norm = adv_norm.clamp(-3.0, 3.0)
weights = (adv_norm + 1.0).clamp(min=0.1, max=5.0)  # Shift to positive

# Step 4: Teacher-forced training
# Concatenate prompt + completion, mask prompt tokens
input_ids, labels = build_concat_with_labels(prompts, generated_texts)
outputs = policy.model(input_ids, labels=None)
per_ex_ce_loss = compute_per_example_ce_loss(outputs.logits, labels)  # [B]

# Step 5: Weight each example's loss
weighted_losses = per_ex_ce_loss * weights
loss = weighted_losses.mean()
```

**new_ragen Algorithm**:
```python
# Step 1: Collect trajectories (rollouts)
trajectory_data = collect_trajectory(temperature)

# Step 2: Compute log probs during generation
for obs, action in zip(observations, actions):
    log_prob = policy.compute_log_probs(instruction, obs, action, requires_grad=True)
    total_log_prob += log_prob

# Step 3: Compute advantages
advantage = total_shaped_reward - v_star

# Step 4: Normalize and clip advantages
adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
adv_clipped = adv_norm.clamp(-5.0, 5.0)

# Step 5: Policy gradient loss
policy_loss = -(log_probs * adv_clipped).mean() * beta
```

---

### 2. V* Computation Method

| Aspect | TinyZero | new_ragen |
|--------|----------|-----------|
| **V* Formula** | Smooth max: `V* = max_r + β*log(mean(exp((r - max_r)/β)))` | Simple max: `V* = max(rewards)` |
| **Beta Used** | Yes (0.5) | No |
| **Rationale** | Smooth approximation of max, more stable | Hard maximum |

**TinyZero V* Code**:
```python
if self.beta > 0:
    max_r = rewards.max()
    exp_terms = np.exp((rewards - max_r) / self.beta)
    V_star = float(max_r + self.beta * np.log(np.mean(exp_terms)))
else:
    V_star = float(rewards.max())
```

**new_ragen V* Code**:
```python
# In stage1_offline.py
def _compute_v_star(self, rewards):
    """Compute V* from list of rewards."""
    if not rewards:
        return 0.0
    return max(rewards)
```

---

### 3. Reward System

| Aspect | TinyZero | new_ragen |
|--------|----------|-----------|
| **Training Rewards** | Partial credit rewards | Shaped rewards (step penalties) |
| **V* Rewards** | Binary rewards (0/1) | Shaped rewards (same as training) |
| **Function** | `compute_reward_with_partial_credit(text, problem, check_reasoning=True)` | `env.step()` returns shaped reward |

**Impact**: TinyZero gives credit for partially correct solutions (e.g., correct reasoning but wrong answer), while new_ragen only rewards final success.

---

### 4. Advantage Weighting

| Aspect | TinyZero | new_ragen |
|--------|----------|-----------|
| **Normalization** | Yes, mean=0, std=1 | Yes, mean=0, std=1 |
| **Clipping Range** | [-3.0, 3.0] | [-5.0, 5.0] |
| **Weight Conversion** | `weights = (adv_norm + 1.0).clamp(0.1, 5.0)` | Uses advantages directly in policy gradient |
| **Weight Schemes** | 3 schemes: 'exp', 'normalized_advantage', 'shifted_advantage' | None (direct policy gradient) |

**TinyZero Weighting Schemes**:
```python
if self.weighting_scheme == 'exp':
    # Original A* weighting (can be unstable)
    weights = torch.exp(advantages / (self.beta + 1e-8))
    weights = weights / weights.mean().clamp_min(1e-6)
elif self.weighting_scheme == 'shifted_advantage':
    weights = (adv_norm + self.adv_clip)  # Shifts to [0, 2*adv_clip]
else:  # 'normalized_advantage' (DEFAULT)
    weights = (adv_norm + 1.0).clamp(min=0.1, max=5.0)
```

---

### 5. KL Divergence Regularization

| Aspect | TinyZero | new_ragen |
|--------|----------|-----------|
| **Has KL Term** | Yes | No |
| **Reference Model** | Required | Not used |
| **KL Coefficient** | 0.02 (default) | N/A |
| **Formula** | `loss = CE_loss + kl_coef * KL(policy \|\| ref)` | N/A |

**TinyZero KL Code**:
```python
# Compute KL divergence per example
if self.kl_coef > 0.0:
    ref_outputs = self.ref_model.model(input_ids, attention_mask)
    ref_logits = ref_outputs.logits.detach()
    kl_per_ex = self._compute_kl_loss(policy_logits, ref_logits, labels)
    per_ex_loss_with_kl = per_ex_ce_loss + self.kl_coef * kl_per_ex
```

---

### 6. Adaptive V* Sampling

| Aspect | TinyZero | new_ragen |
|--------|----------|-----------|
| **Adaptive Sampling** | Yes | No |
| **Early Training** | 5 samples per instruction | Fixed k_samples |
| **Mid Training** | 3 samples per instruction | Fixed k_samples |
| **Late Training** | 2 samples per instruction | Fixed k_samples |
| **V* Caching** | Yes, with prompt-based lookup | Yes, instruction-based |

**TinyZero Adaptive Code**:
```python
if self.adaptive_vstar:
    if self.step < 30:
        num_samples = 5  # Early: need accurate V*
    elif self.step < 70:
        num_samples = 3  # Mid: medium accuracy
    else:
        num_samples = 2  # Late: model is trained
```

---

### 7. Training Mechanics

| Aspect | TinyZero | new_ragen |
|--------|----------|-----------|
| **Token Masking** | Mask prompt tokens, only train on completion | Train on full sequence |
| **Per-Example Loss** | Yes, computed per example then weighted | No, trajectory-level loss |
| **Gradient Accumulation** | No (not needed with per-example weighting) | Yes (to avoid OOM) |
| **Learning Rate** | 5e-7 | 5e-6 |

---

### 8. Hyperparameters

| Parameter | TinyZero | new_ragen |
|-----------|----------|-----------|
| `beta` | 0.5 (used in V* and weighting) | 0.5 (used as loss scaling) |
| `learning_rate` | 5e-7 | 5e-6 |
| `adv_clip` | 3.0 | 5.0 |
| `temperature` | 0.8 | 0.8 |
| `kl_coef` | 0.02 | N/A |
| `clip_grad_norm` | 1.0 | 1.0 |

---

## Why TinyZero Works and new_ragen Doesn't

### 1. **Weighted Supervised Learning is More Stable**
   - TinyZero trains like SFT but upweights good examples, downweights bad ones
   - new_ragen uses high-variance policy gradient (REINFORCE)
   - Supervised learning converges faster and more reliably

### 2. **Partial Credit Rewards**
   - TinyZero gives credit for progress (reasoning steps, partial correctness)
   - new_ragen only rewards final success (binary: buy correct product or fail)
   - Provides denser learning signal

### 3. **Teacher-Forcing on Completions**
   - TinyZero: Generate once, then train on that exact text multiple times
   - new_ragen: Uses log probs computed during generation (less stable)
   - Teacher-forcing leverages full context

### 4. **Positive Weights Only**
   - TinyZero: Converts advantages to positive weights [0.1, 5.0]
   - new_ragen: Uses raw advantages (can be negative)
   - Avoids "unlearning" bad examples (still trains on them, just with lower weight)

### 5. **KL Regularization**
   - TinyZero: Keeps policy close to reference model (prevents mode collapse)
   - new_ragen: No regularization
   - Helps maintain diversity and prevent overfitting to high-reward mode

---

## Implications for new_ragen

### The Current Implementation is NOT A*-PO

new_ragen implements **REINFORCE with advantage baseline**, not the weighted supervised learning approach from TinyZero.

### Options Moving Forward

#### Option A: Keep REINFORCE, Debug Issues
- Add partial credit rewards
- Improve exploration (curriculum learning)
- Add KL regularization
- Better advantage normalization
- **Pros**: True policy gradient, theoretically sound
- **Cons**: High variance, needs more tuning

#### Option B: Switch to TinyZero's Weighted SFT Approach (RECOMMENDED)
- Generate completions from policy
- Compute advantages
- Convert to weights: `weights = (adv_norm + 1.0).clamp(0.1, 5.0)`
- Train with weighted CE loss on completions
- Add KL term with reference model
- **Pros**: Proven to work, more stable, faster convergence
- **Cons**: Need reference model, more complex implementation

---

## Recommended Next Steps

### 1. **Implement TinyZero's Weighted SFT Approach**

This is the most promising path since it's proven to work with the same model.

**Key Changes Needed**:
1. Modify `stage2_online.py` to use teacher-forcing instead of REINFORCE
2. Implement per-example CE loss computation
3. Add advantage-to-weight conversion: `(adv_norm + 1.0).clamp(0.1, 5.0)`
4. Add KL divergence term (optional but recommended)
5. Change V* to use smooth max with beta

### 2. **Add Partial Credit Rewards**

WebShop needs intermediate rewards:
- Reward for good search queries
- Reward for clicking relevant products
- Reward for being on correct product page
- Final reward for correct purchase

### 3. **Implement Curriculum Learning**

Start with easier tasks (fewer products, clearer instructions), gradually increase difficulty.

### 4. **Adaptive V* Sampling**

Reduce V* computation as training progresses (5 → 3 → 2 samples).

---

## Code Architecture Comparison

### TinyZero Structure:
```
tinyzero/
├── apo_trainer.py        # Main trainer with weighted SFT
├── vstar_cache.py        # V* caching system
├── rewards.py            # Partial credit reward computation
└── [other files]
```

### new_ragen Structure:
```
new_ragen/
├── train_apo.py          # Main training script
├── src/
│   ├── trainers/
│   │   ├── stage1_offline.py   # V* computation
│   │   └── stage2_online.py    # Policy optimization
│   ├── models/
│   │   └── policy.py           # Policy model
│   ├── environments/
│   │   └── webshop_env.py      # Environment wrapper
│   └── utils/
│       └── value_cache.py      # V* caching
```

---

## Conclusion

**The fundamental issue is algorithmic, not bugs**: new_ragen implements a different algorithm than TinyZero.

To make new_ragen work, you need to either:
1. **Switch to TinyZero's weighted SFT approach** (recommended)
2. OR **Debug and improve the REINFORCE implementation** with:
   - Partial credit rewards
   - Curriculum learning
   - Better exploration strategies
   - KL regularization

Given that TinyZero's approach is proven to work with the same model, **Option 1 is strongly recommended**.
