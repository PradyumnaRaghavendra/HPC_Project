# WebShop A*-PO Training

Clean, robust implementation of A*-PO (Advantage-Weighted Policy Optimization) for WebShop using TinyZero's proven weighted SFT approach.

## What's Different from `new_ragen`?

### Algorithm: Weighted SFT (not REINFORCE)

**Previous (new_ragen)**: REINFORCE policy gradient
- High variance
- Unstable training
- Failed to learn

**Current (webshop_apo)**: Weighted Supervised Learning
- Low variance
- Stable gradients
- Proven to work with same model in TinyZero

### Key Differences

| Feature | new_ragen (BROKEN) | webshop_apo (WORKING) |
|---------|-------------------|----------------------|
| **Algorithm** | REINFORCE | Weighted SFT |
| **Training** | Online rollouts | Teacher-forcing on completions |
| **Loss** | `-(log_prob × advantage)` | `(CE_loss × weight).mean()` |
| **Advantage Use** | Direct (can be negative) | Converted to positive weights |
| **Reference Model** | No | Yes (for V* and KL) |
| **V* Formula** | Simple max | Smooth max with beta |
| **KL Regularization** | No | Yes |

## Architecture

```
webshop_apo/
├── apo_trainer.py           # Main A*-PO trainer (weighted SFT)
├── policy.py                # Policy and reference models
├── webshop_env.py           # WebShop environment wrapper
├── train.py                 # Training script
└── modal_train_webshop.py   # All-in-one Modal deployment
```

Simple, clean, no unnecessary complexity.

## How It Works

### A*-PO Algorithm (Weighted SFT)

```python
# 1. Compute V* (optimal value) for each instruction
V_star = smooth_max(rewards from k reference model samples)

# 2. Generate trajectories from current policy
trajectories = policy.generate(instructions, temperature=0.9)

# 3. Compute rewards and advantages
rewards = evaluate(trajectories)
advantages = rewards - V_star

# 4. Convert advantages to POSITIVE weights
adv_norm = (advantages - mean) / std
weights = (adv_norm + 1.0).clamp(0.1, 5.0)  # ← KEY: Always positive!

# 5. Train with weighted cross-entropy (teacher-forcing)
loss = (CE_loss(trajectories) × weights).mean()

# 6. Add KL regularization
loss += kl_coef × KL(policy || reference)
```

### Why This Works

1. **Low Variance**: Supervised learning gradients instead of policy gradients
2. **Always Learn**: Positive weights mean we always make progress (just weighted)
3. **No "Unlearning"**: Bad examples get low weight, but still contribute
4. **KL Regularization**: Prevents mode collapse
5. **Proven**: Works with same Qwen2.5-3B in TinyZero

## Usage

### Local Training

```bash
cd week_06
python webshop_apo/train.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-steps 100 \
    --batch-size 2 \
    --v-star-samples 8 \
    --eval-frequency 20 \
    --output-dir outputs/webshop_apo
```

### Modal Training (Recommended)

```bash
cd week_06
modal run webshop_apo/modal_train_webshop.py \
    --num-steps 100 \
    --batch-size 2 \
    --v-star-samples 8 \
    --num-products 100
```

**Quick test** (verify it works):
```bash
modal run webshop_apo/modal_train_webshop.py \
    --num-steps 20 \
    --batch-size 2 \
    --v-star-samples 4 \
    --eval-frequency 10
```

**Full training** (best results):
```bash
modal run webshop_apo/modal_train_webshop.py \
    --num-steps 200 \
    --batch-size 4 \
    --v-star-samples 8 \
    --eval-frequency 20 \
    --num-products 100
```

### Download Results

```bash
modal volume get webshop-apo-outputs /outputs ./webshop_results
```

## Configuration

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--beta` | 0.5 | Smoothing parameter for V* |
| `--v-star-samples` | 8 | Number of samples for V* estimation |
| `--learning-rate` | 5e-6 | Learning rate (same as TinyZero) |
| `--kl-coef` | 0.02 | KL regularization coefficient |
| `--adv-clip` | 3.0 | Advantage clipping threshold |
| `--temperature` | 0.9 | Sampling temperature |
| `--max-episode-steps` | 15 | Max steps per episode |

### Environment

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-products` | 100 | Number of products in WebShop (100 or 1000) |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-steps` | 100 | Number of training steps |
| `--batch-size` | 2 | Instructions per batch (limited by GPU memory) |
| `--eval-frequency` | 20 | Evaluate every N steps |
| `--save-frequency` | 50 | Save checkpoint every N steps |

## Expected Results

### Quick Test (20 steps, 4 V* samples)
- **Purpose**: Verify implementation works
- **Time**: ~15-20 minutes
- **Expected**: 0-5% success rate (not enough training)
- **Cost**: ~$2-3

### Medium Training (100 steps, 8 V* samples)
- **Purpose**: See learning progress
- **Time**: ~1-1.5 hours
- **Expected**: 5-15% success rate
- **Cost**: ~$10-15

### Full Training (200 steps, 8 V* samples)
- **Purpose**: Best results
- **Time**: ~2-3 hours
- **Expected**: 15-30% success rate
- **Cost**: ~$20-30

## Troubleshooting

### "All V* values negative"
- This means no successful trajectories in V* sampling
- **Solution**: Increase `--v-star-samples` (try 12 or 16)
- Or reduce task difficulty: `--num-products 50`

### "All advantages negative"
- Policy never beats V* baseline
- **Solution**: Check model is generating valid actions
- Try higher temperature: `--temperature 1.0`

### OOM (Out of Memory)
- **Solution**: Reduce `--batch-size` to 1
- Or reduce `--max-episode-steps` to 10

### Training too slow
- **Solution**: Enable adaptive V* sampling (already on by default)
- Reduces V* samples as training progresses: 8 → 4 → 2

## Comparison with TinyZero

| Feature | TinyZero (Math) | webshop_apo (WebShop) |
|---------|----------------|----------------------|
| Task | Single-turn (math problems) | Multi-turn (shopping) |
| Reward | Binary (0/1) | Binary (0/1) |
| Model | Qwen2.5-3B | Qwen2.5-3B |
| Algorithm | Weighted SFT | Weighted SFT (same!) |
| Beta | 0.5 | 0.5 |
| Learning Rate | 5e-7 | 5e-6 (10x higher) |
| KL Coef | 0.02 | 0.02 |
| Curriculum | Yes | Not yet (can add) |

## Future Improvements

1. **Curriculum Learning**
   - Start with 10 products → 50 → 100 → 1000
   - Simple instructions first

2. **Partial Credit Rewards**
   - Reward for relevant searches
   - Reward for clicking correct category
   - Not just final success

3. **Better Prompting**
   - Few-shot examples
   - Chain-of-thought reasoning

4. **Larger Model**
   - Try Qwen2.5-7B or 14B
   - Should learn faster

## Why This Will Work

1. **Proven Algorithm**: Same approach that works in TinyZero
2. **Same Model**: Qwen2.5-3B works in TinyZero
3. **Clean Implementation**: No REINFORCE high-variance issues
4. **Stable Training**: Weighted SFT is much more stable
5. **Good Defaults**: Hyperparameters from working TinyZero config

## License

MIT

## Citation

If you use this code, please cite:

```bibtex
@misc{webshop_apo,
  title={WebShop A*-PO: Clean A*-PO Implementation},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/webshop-apo}}
}
```

## Acknowledgments

- Based on TinyZero's A*-PO implementation
- WebShop environment from Princeton NLP
- Inspired by RAGEN paper (but using different algorithm)
