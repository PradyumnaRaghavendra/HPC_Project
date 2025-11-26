# A* APO Implementation - Critical Fixes Applied

## Summary
Your A* APO implementation had several critical issues causing poor training performance. The main symptom was: loss dropping to near-zero while performance degraded (peaked at 80%, dropped to 40%). This document details all issues found and fixes applied.

---

## Critical Issues Fixed

### 1. **INCORRECT A* APO LOSS FUNCTION** ⚠️ (Most Critical)

**Location**: `src/trainers/stage2_online.py:242-246`

**Original (Wrong) Code**:
```python
targets = (self.beta * advantages_tensor).detach()
loss = ((log_probs_tensor - targets) ** 2).mean()
```

**Problem**:
- Treated A* APO as **regression** problem (MSE loss)
- Tried to make log_prob equal scaled advantage
- Log probs are negative (-5 to -20), advantages can be any value
- No theoretical justification for this formulation
- Caused mode collapse and reward hacking

**Fixed Code**:
```python
# Normalize advantages
advantages_normalized = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

# Clip to prevent extreme updates
advantages_clipped = torch.clamp(advantages_normalized, -5.0, 5.0)

# Policy gradient loss: maximize log_prob weighted by advantage
policy_loss = -(log_probs_tensor * advantages_clipped).mean() * self.beta
```

**Why This Fix Works**:
- Uses proper **policy gradient** formulation: ∇L = -E[log π(y|x) * A*(x,y)]
- Increases probability of actions with positive advantage
- Decreases probability of actions with negative advantage
- Advantage normalization improves stability
- Advantage clipping prevents extreme updates from poor V* estimates

---

### 2. **Missing Environment Method**

**Location**: `src/environments/webshop_env.py`

**Problem**: Stage 1 tried to use `env.reset_with_instruction()` which didn't exist. This meant all k=64 trajectory samples for each instruction used the SAME initial observation, leading to poor V* estimates.

**Fix**: Added `reset_with_instruction()` method that:
- Resets environment multiple times to find the target instruction
- Returns correct initial observation for each trajectory
- Provides diverse trajectories for better V* estimation

---

### 3. **Poor Hyperparameters**

**Changes Made**:

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `learning_rate` | 1e-5 | 5e-6 | More stable training with policy gradient |
| `beta` | 0.1 | 0.5 | Stronger advantage signal |
| `temperature` | 1.0 | 0.8 | More focused exploration |

---

### 4. **No Advantage Normalization**

**Fix**: Added advantage normalization and clipping
- Normalizes to mean=0, std=1 for stable gradients
- Clips to [-5, 5] to prevent extreme updates
- Helps when V* estimates are inaccurate

---

### 5. **Improved Logging**

Added metrics to track training health:
- `mean_advantage`: Raw advantage values
- `std_advantage`: Advantage variance
- `mean_advantage_normalized`: After normalization
- `advantages_clipped_ratio`: How often clipping occurs

---

## How to Use the Fixed Code

### Training from Scratch

```bash
cd new_ragen

# Stage 1 + Stage 2 (recommended)
python train_apo.py \
    --num_products 100 \
    --num_instructions_stage1 50 \
    --k_samples 64 \
    --num_iterations 500 \
    --learning_rate 5e-6 \
    --beta 0.5 \
    --temperature 0.8

# Skip Stage 1 if you have V* cache
python train_apo.py --skip_stage1 --num_iterations 500
```

### Recommended Settings

**For Quick Testing** (faster, less accurate):
```bash
python train_apo.py \
    --num_products 100 \
    --num_instructions_stage1 20 \
    --k_samples 32 \
    --num_iterations 200
```

**For Better Performance** (slower, more accurate):
```bash
python train_apo.py \
    --num_products 100 \
    --num_instructions_stage1 100 \
    --k_samples 64 \
    --num_iterations 1000 \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct"  # Larger model
```

---

## What to Expect Now

### Stage 1 (Offline V* Estimation)
- Should see diverse reward distributions for each instruction
- V* values should be reasonable (typically 0.5-1.5 for WebShop)
- Progress: ~50 instructions with k=64 samples each

### Stage 2 (Online Training)
**Good Training Signals**:
- Loss should be negative (we're maximizing)
- Loss magnitude should gradually increase (learning better actions)
- Mean advantage should increase over time
- Success rate should improve steadily

**Warning Signs**:
- Loss stays near zero → Not learning
- Advantages all near zero → V* estimates are too optimistic
- High clipping ratio (>0.5) → Consider adjusting advantage clipping range

---

## Additional Recommendations

### 1. Model Size
Current: `Qwen/Qwen2.5-0.5B-Instruct` (500M params)
- **Too small for complex WebShop tasks**
- Recommend: `Qwen/Qwen2.5-1.5B-Instruct` or larger
- Will learn faster and achieve better performance

### 2. Batch Size
Current: 8 episodes per update
- Increase to 16 or 32 if you have GPU memory
- Larger batches = more stable gradients

### 3. Stage 1 Diversity
- Increase `k_samples` to 128 for better V* estimates
- Increase `num_instructions_stage1` to 100 for better coverage

### 4. Monitoring Training
Use TensorBoard to visualize:
```bash
tensorboard --logdir logs/
```

Watch these metrics:
- `train/loss` - should become more negative
- `train/mean_advantage` - should increase
- `eval/success_rate` - should improve
- `eval/mean_shaped_reward` - should increase

---

## Theory: Why A* APO Works

A* APO (Adaptive Advantage-based Policy Optimization) combines:

1. **Offline V* Estimation (Stage 1)**:
   - Sample k trajectories per instruction
   - V*(x) = max reward achieved
   - Acts as optimal baseline

2. **Online Policy Optimization (Stage 2)**:
   - Compute advantage: A*(x,y) = R(x,y) - V*(x)
   - Positive advantage → increase π(y|x)
   - Negative advantage → decrease π(y|x)
   - No need for value function training!

**Advantages over PPO**:
- No value function needed (uses V* cache instead)
- More stable (optimal baseline reduces variance)
- More sample efficient (learns from both good and bad trajectories)

---

## Debugging Tips

### If training is slow/not improving:

1. **Check V* values**:
   ```python
   # Load and inspect
   import json
   with open('checkpoints/v_star_cache.json', 'r') as f:
       cache = json.load(f)

   # Check statistics
   values = list(cache.values())
   print(f"Mean V*: {np.mean(values)}")
   print(f"Std V*: {np.std(values)}")
   ```

2. **Monitor advantages during training**:
   - Should see both positive and negative values
   - If all near zero: V* too optimistic (increase k_samples in Stage 1)
   - If too large (>10): V* too pessimistic (check reward shaping)

3. **Check action generation**:
   - Model should generate valid actions: `search[...]`, `click[...]`, `buy now`
   - If generating gibberish: reduce temperature or increase model size

---

## Files Modified

1. ✅ `src/trainers/stage2_online.py` - Fixed loss function, added normalization/clipping
2. ✅ `src/trainers/stage1_offline.py` - Fixed trajectory sampling
3. ✅ `src/environments/webshop_env.py` - Added reset_with_instruction()
4. ✅ `train_apo.py` - Updated default hyperparameters

---

## ⚠️ CRITICAL: Minimum Requirements for Good Training

### **DO NOT use the "quick test" settings for real training!**

The quick test (5 instructions, 4 samples) will NOT learn anything useful. It proved the code works, but you need:

### **Minimum Settings for Learning**:

```bash
modal run modal_train.py \
    --num-instructions-stage1 50   # Minimum! 100+ is better
    --k-samples 16                  # Minimum! 32+ is better
    --num-iterations 200            # Minimum!
    --batch-size 4 \
    --eval-frequency 20
```

**Why these numbers?**
- **50+ instructions**: Need diverse V* estimates across many tasks
- **16+ samples per instruction**: Need to find the BEST trajectory (V*)
- **200+ iterations**: Policy needs time to learn from diverse examples
- **Batch size 4**: Memory-efficient with gradient accumulation

### **Expected Training Time & Cost**:
- Minimum settings: ~2-3 hours, ~$5-8
- Good settings (100 inst, 32 samples, 500 iter): ~5-6 hours, ~$15-20

### **Warning Signs During Training**:

1. **All V* values negative**: No successful trajectories in Stage 1
   - Solution: Increase k_samples or check reward shaping

2. **All advantages negative**: Policy never beats V*
   - Solution: Need more exploration or better V* estimates

3. **Loss magnitude increasing rapidly**: Possible divergence
   - Solution: Reduce learning rate or beta

4. **Success rate stays 0%**: Task is too hard or reward shaping broken
   - Solution: Check environment, try simpler task, or add curriculum

## Next Steps

1. **Run proper Stage 1 first**:
   ```bash
   modal run modal_train.py \
       --num-instructions-stage1 100 \
       --k-samples 32 \
       --num-iterations 1  # Just test Stage 1
   ```

   Check output for V* statistics. You want:
   - Max V* > 0 (some successful trajectories)
   - Mean V* around 0.2-0.5 (mix of success/failure)
   - Std V* > 0 (diversity)

2. **Then run full training**:
   ```bash
   modal run modal_train.py \
       --num-instructions-stage1 100 \
       --k-samples 32 \
       --num-iterations 300 \
       --batch-size 4 \
       --eval-frequency 25
   ```

3. **Monitor for**:
   - At least SOME positive advantages in batches
   - Success rate gradually increasing (even 5% → 10% is progress!)
   - Loss should be negative and relatively stable

4. **If still struggling after 100 iterations**:
   - Try smaller num_products (100 instead of 1000)
   - Increase temperature for more exploration
   - Check if any trajectories succeed manually

---

## Contact / Questions

If you encounter issues:
1. Check training logs for errors
2. Verify V* cache has reasonable values
3. Monitor advantages during training (should be diverse, not all zero)
4. Try different hyperparameters (see recommendations above)

Good luck with training! The fixes address the fundamental algorithmic errors, so you should see much better performance now. 🚀
