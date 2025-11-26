"""
A*-PO-style Training on Modal with WebShop + Qwen2.5-3B-Instruct

This is a single-file, self-contained Modal script.
It combines all the corrected logic from the modular files
(policy, stage1, stage2, etc.) into one file that is
compatible with older Modal library versions that do not
support file mounting.

VERSION 3: Adds robust timeouts to all env.reset() and env.step()
calls to prevent the script from hanging.
"""

import modal

# --------------------------------------------------------------------------------
# Image & Environment Setup
# (This block is unchanged from your working version)
# --------------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "gnupg",
        "software-properties-common",
        "ca-certificates",
        "curl",
    )
    .run_commands(
        # Install Temurin JDK 21
        "mkdir -p /etc/apt/keyrings",
        "wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc",
        "echo 'deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb bookworm main' | tee /etc/apt/sources.list.d/adoptium.list",
        "apt-get update",
        "apt-get install -y temurin-21-jdk",
    )
    .pip_install(
        # Core
        "numpy>=1.25.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "accelerate>=0.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
        # Retrieval / search
        "faiss-cpu>=1.7.0",
        "pyserini>=0.39.0",
        "rank-bm25",
        # WebShop + utils
        "gym==0.24.0",
        "spacy>=3.6.0",
        "flask==2.1.2",
        "Werkzeug==2.0.3",
        "beautifulsoup4",
        "nltk",
        "cleantext",
        "requests",
        "selenium",
        "gdown",
        "pandas",
        "scikit-learn",
        "rich",
        "thefuzz",
        "python-Levenshtein",
    )
    .run_commands(
        # Clone WebShop
        "git clone https://github.com/princeton-nlp/WebShop.git /opt/WebShop",
        "mkdir -p /opt/WebShop/data",
        # Download ALL data files for more goals
        "cd /opt/WebShop/data && python -m gdown 1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib -O items_shuffle_1000.json",
        "cd /opt/WebShop/data && python -m gdown 1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu -O items_ins_v2_1000.json",
        "cd /opt/WebShop/data && python -m gdown 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O -O items_human_ins.json",
        # SpaCy model
        "python -m spacy download en_core_web_sm",
    )
)

app = modal.App("apo-webshop-trainer", image=image)
volume = modal.Volume.from_name("ragen-storage", create_if_missing=True)

# --------------------------------------------------------------------------------
# Remote Training Function
# --------------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/data": volume},
)
def run_training_on_modal(config: dict):
    """
    This is the remote function that runs our modular training code.
    All helper classes are defined inside this function scope.
    """

    import os
    import sys
    import json
    import re
    import subprocess
    import gc
    import signal # <-- NEW: Import signal for timeouts
    from pathlib import Path
    from collections import defaultdict
    from typing import Dict, List, Tuple, Optional

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

    # Add WebShop to the Python path
    sys.path.insert(0, "/opt/WebShop")
    from web_agent_site.envs import WebAgentTextEnv

    # ========================================================================
    # HELPER CLASS: Timeout
    # (This is the new robust solution)
    # ========================================================================
    
    class TimeoutError(Exception):
        pass

    class Timeout:
        def __init__(self, seconds=10, error_message='Timeout'):
            self.seconds = seconds
            self.error_message = error_message
        def handle_timeout(self, signum, frame):
            raise TimeoutError(self.error_message)
        def __enter__(self):
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        def __exit__(self, type, value, traceback):
            signal.alarm(0)

    # ========================================================================
    # HELPER CLASS: Utils
    # ========================================================================
    
    def _clean_instruction(text: str) -> str:
        if not isinstance(text, str): return ""
        s = text.strip()
        for _ in range(2):
            low = s.lower()
            if low.startswith("instruction:"): s = s.split(":", 1)[1].strip(); continue
            if low.startswith("goal:"): s = s.split(":", 1)[1].strip(); continue
            break
        return s

    # ========================================================================
    # HELPER CLASS: WebShopEnvironment
    # (Unchanged, the timeouts will be applied when *calling* its methods)
    # ========================================================================

    class WebShopEnvironment:
        """Env wrapper that uses WebShop's own goals."""
        def __init__(self, num_products: int, max_steps: int, human_goals: bool):
            self.env = WebAgentTextEnv(
                observation_mode="text", num_products=num_products, human_goals=human_goals,
            )
            self.max_steps = max_steps; self.step_count = 0; self.action_history = []
            try:
                n_goals = len(getattr(self.env, "goals", []))
                print(f"Loaded {n_goals} WebShop goals in env.", flush=True)
            except Exception as e: print(f"[WARN] Could not inspect goals: {e}", flush=True)
        def reset(self):
            result = self.env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            self.step_count = 0; self.action_history = []
            inst = self._get_instruction_from_env(obs)
            if not inst: print("  [WARN] Failed to extract instruction", flush=True)
            return obs, {"instruction": inst, "step": 0, "done": False}
        def reset_with_instruction(self, target_instruction: str, max_tries: int = 15):
            target_clean = _clean_instruction(target_instruction)
            last_obs, last_info = None, None
            for _ in range(max_tries):
                obs, info = self.reset()
                inst = _clean_instruction(info.get("instruction", ""))
                last_obs, last_info = obs, info
                if inst == target_clean: return obs, info
            return last_obs, last_info
        def step(self, action: str):
            parsed_action = self._parse_action(action)
            self.action_history.append(parsed_action); self.step_count += 1
            obs, reward, done, info = self.env.step(parsed_action)
            if info is None: info = {}
            shaped = self._shape_reward(float(reward), parsed_action, bool(done))
            out = {"raw_reward": float(reward), "shaped_reward": shaped, "action_type": self._get_action_type(parsed_action), "step": self.step_count}
            out.update(info); return obs, shaped, bool(done), out
        def _get_instruction_from_env(self, obs: str) -> str:
            if hasattr(self.env, "instruction_text"):
                val = getattr(self.env, "instruction_text");
                if isinstance(val, str) and val.strip(): return _clean_instruction(val)
            if hasattr(self.env, "state") and isinstance(self.env.state, dict):
                cand = (self.env.state.get("instruction_text") or self.env.state.get("goal") or "");
                if isinstance(cand, str) and cand.strip(): return _clean_instruction(cand)
            if isinstance(obs, str):
                m = re.search(r"(Instruction|Goal):\s*(.+?)(?:\n|$)", obs, flags=re.IGNORECASE);
                if m: return _clean_instruction(m.group(2))
            return ""
        def _shape_reward(self, base_reward: float, action: str, done: bool) -> float:
            reward = base_reward; action_type = self._get_action_type(action)
            if action_type == "search": reward += 0.05
            elif action_type == "click": reward += 0.10
            elif action_type == "buy": reward += 0.50 if base_reward > 0 else -0.20
            reward -= 0.01 * self.step_count
            if action_type == "search" and self.action_history.count(action) > 2: reward -= 0.05
            return float(reward)
        def _parse_action(self, action: str) -> str:
            raw, low = str(action).strip(), str(action).strip().lower()
            if low == "buy now": return "buy now"
            if low.startswith("search[") and low.endswith("]"): return raw
            if low.startswith("click[") and low.endswith("]"): return raw
            return f"search[{raw}]"
        def _get_action_type(self, action: str) -> str:
            a = str(action).strip().lower()
            if a.startswith("search["): return "search"
            if a.startswith("click["): return "click"
            if a == "buy now": return "buy"
            return "unknown"

    # ========================================================================
    # HELPER CLASS: RAGENPolicy
    # (Unchanged)
    # ========================================================================

    class RAGENPolicy(nn.Module):
        def __init__(self, model_name: str, max_length: int, device: str):
            super().__init__(); self.model_name = model_name; self.max_length = max_length; self.device = device
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            tok_kwargs = {"token": hf_token} if hf_token else {}
            model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16, "device_map": device} if hf_token else {"torch_dtype": torch.bfloat16, "device_map": device}
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = len(self.tokenizer)
        def format_prompt(self, instruction: str, observation: str, history: List[Tuple[str, str]] = None) -> str:
            obs_truncated = observation[:1000]; return f"You are a shopping agent. Your task is to find and purchase items.\n\nInstruction: {instruction}\n\nCurrent Page:\n{obs_truncated}\n\nYou can take these actions:\n- search[query] - Search for products\n- click[item_id] - Click on an item to view details\n- buy now - Purchase the current item\n\nGenerate ONLY the next action, nothing else.\nAction:"
        def generate_action(self, instruction: str, observation: str, history: List[Tuple[str, str]] = None, temperature: float = 1.0, sample: bool = True) -> Tuple[str, Dict]:
            prompt = self.format_prompt(instruction, observation, history)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=50, temperature=max(temperature, 0.1), do_sample=sample, 
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_scores=True
                )
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            action = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            action = self._clean_action(action); info = {"prompt_length": inputs.input_ids.shape[1], "generated_length": len(generated_ids), "raw_output": action}
            return action, info
        def _clean_action(self, action: str) -> str:
            action = action.strip();
            if "\n" in action: action = action.split("\n")[0]
            for delimiter in [".", ",", "Explanation:", "Reason:"]:
                if delimiter in action: action = action.split(delimiter)[0]
            return action.strip()
        def compute_log_probs(self, instruction: str, observation: str, action: str, history: List[Tuple[str, str]] = None, requires_grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
            prompt = self.format_prompt(instruction, observation, history); full_text = prompt + " " + action
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            with torch.set_grad_enabled(requires_grad): outputs = self.model(**inputs); logits = outputs.logits
            prompt_len = prompt_inputs.input_ids.shape[1]
            if inputs.input_ids.shape[1] <= prompt_len: return torch.tensor(0.0, device=self.device, requires_grad=requires_grad), torch.tensor(0.0, device=self.device)
            action_logits = logits[0, prompt_len-1:-1]; action_ids = inputs.input_ids[0, prompt_len:]
            log_probs_dist = torch.nn.functional.log_softmax(action_logits, dim=-1)
            action_log_probs = log_probs_dist.gather(1, action_ids.unsqueeze(1)).squeeze(1); total_log_prob = action_log_probs.sum()
            probs_dist = torch.nn.functional.softmax(action_logits, dim=-1); entropy = -(probs_dist * log_probs_dist).sum(dim=-1).mean()
            return total_log_prob, entropy
        def save(self, path: str):
            Path(path).mkdir(parents=True, exist_ok=True); self.model.save_pretrained(path); self.tokenizer.save_pretrained(path)

    # ========================================================================
    # HELPER CLASS: OptimalValueCache
    # (Unchanged)
    # ========================================================================

    class OptimalValueCache:
        def __init__(self, cache_file: str = "v_star_cache.json"):
            self.cache_file = Path(cache_file); self.v_star: Dict[str, float] = {};
            if self.cache_file.exists(): self.load()
        def update(self, instruction: str, reward: float):
            instruction = instruction.strip();
            if not instruction: return
            # Ensure reward is always a float (not dict or other type)
            try:
                reward_float = float(reward)
            except (TypeError, ValueError):
                print(f"[WARN] Invalid reward type: {type(reward)}, skipping update", flush=True)
                return
            current_best = self.v_star.get(instruction, -float('inf'));
            if reward_float > current_best: self.v_star[instruction] = reward_float
        def get(self, instruction: str, default: float = 0.0) -> float:
            instruction = instruction.strip();
            if not instruction: return default
            return self.v_star.get(instruction, default)
        def has(self, instruction: str) -> bool: return instruction in self.v_star
        def save(self):
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f: json.dump(self.v_star, f, indent=2)
        def load(self):
            try:
                with open(self.cache_file, 'r') as f: self.v_star = json.load(f)
            except json.JSONDecodeError: self.v_star = {}
        def get_statistics(self) -> Dict:
            if not self.v_star: return {"num_instructions": 0, "mean_v_star": 0.0, "max_v_star": 0.0, "min_v_star": 0.0}
            # Filter out any non-float values that might have been saved
            values = [float(v) for v in self.v_star.values() if isinstance(v, (int, float))]
            if not values: return {"num_instructions": 0, "mean_v_star": 0.0, "max_v_star": 0.0, "min_v_star": 0.0}
            return {"num_instructions": len(self.v_star), "mean_v_star": np.mean(values), "max_v_star": np.max(values), "min_v_star": np.min(values), "std_v_star": np.std(values)}

    # ========================================================================
    # HELPER CLASS: Stage1OfflineTrainer
    # (MODIFIED with timeouts)
    # ========================================================================

    class Stage1OfflineTrainer:
        def __init__(self, env, policy, value_cache: OptimalValueCache, k_samples: int, max_steps: int, temperature: float):
            self.env = env; self.policy = policy; self.value_cache = value_cache;
            self.k_samples = k_samples; self.max_steps = max_steps; self.temperature = temperature
        
        def collect_optimal_values(self, num_instructions: int, verbose: bool = True):
            print(f"\n{'='*60}\nStage 1: Offline Value Estimation\n{'='*60}")
            print(f"Collecting {self.k_samples} trajectories per instruction")
            print(f"Target instructions: {num_instructions}\n{'='*60}\n")
            
            instruction_count = 0; instructions_seen = set()
            pbar = tqdm(total=num_instructions, desc="Instructions processed")
            
            while instruction_count < num_instructions:
                try:
                    # --- NEW: Added 60 second timeout to reset ---
                    with Timeout(seconds=60, error_message="env.reset() timed out"):
                        obs, info = self.env.reset()
                except TimeoutError as e:
                    print(f"\n[WARN] {e}. Skipping.", flush=True)
                    continue # Skip this entire iteration
                except Exception as e:
                    print(f"\n[WARN] env.reset() failed: {e}. Skipping.", flush=True)
                    continue

                instruction = info.get("instruction", "")
                if not instruction or instruction in instructions_seen: continue
                
                instructions_seen.add(instruction); rewards = []
                for k in range(self.k_samples):
                    try:
                        # --- NEW: Added 60 second timeout to reset_with_instruction ---
                        with Timeout(seconds=60, error_message="env.reset_with_instruction() timed out"):
                            obs_k = obs if k == 0 else self.env.reset_with_instruction(instruction)[0]
                    except TimeoutError as e:
                        print(f"\n[WARN] {e}. Skipping sample k={k}.", flush=True)
                        continue # Skip this sample
                    except Exception as e:
                        print(f"\n[WARN] env.reset_with_instruction() failed: {e}. Skipping sample k={k}.", flush=True)
                        continue
                    
                    # Trajectory itself is now robust with internal timeouts
                    reward = self._run_single_trajectory(instruction, obs_k)
                    rewards.append(reward)
                
                if not rewards: # If all samples failed
                    print(f"\n[WARN] All samples failed for instruction. Skipping.", flush=True)
                    instruction_count += 1; pbar.update(1); continue

                v_star = max(rewards); self.value_cache.update(instruction, v_star)
                
                if verbose and (instruction_count + 1) % 10 == 0:
                    print(f"\nInstruction: {instruction[:50]}...", flush=True)
                    print(f"  Rewards: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={np.mean(rewards):.3f}", flush=True)
                    print(f"  V*: {v_star:.3f}", flush=True)
                
                instruction_count += 1; pbar.update(1)
            
            pbar.close(); self.value_cache.save(); stats = self.value_cache.get_statistics()
            print(f"\n{'='*60}\nStage 1 Complete!\n{'='*60}")
            print(f"Instructions processed: {stats['num_instructions']}")
            print(f"Mean V*: {stats['mean_v_star']:.3f}")
            print(f"Max V*: {stats['max_v_star']:.3f}")
            print(f"Min V*: {stats['min_v_star']:.3f}")
            print(f"Std V*: {stats.get('std_v_star', 0.0):.3f}")
            print(f"{'='*60}\n", flush=True)

            # WARNING: Check if all V* are negative (bad sign!)
            if stats['max_v_star'] < 0:
                print("⚠️  WARNING: All V* values are negative!")
                print("⚠️  This means NO successful trajectories in Stage 1.")
                print("⚠️  Training will struggle without positive examples.\n", flush=True)
        
        def _run_single_trajectory(self, instruction: str, initial_obs: str) -> float:
            obs = initial_obs; total_shaped_reward = 0.0; done = False; step = 0
            while not done and step < self.max_steps:
                try:
                    action, _ = self.policy.generate_action(
                        instruction=instruction, observation=obs,
                        temperature=self.temperature, sample=True
                    )
                    # --- NEW: Added 30 second timeout to step ---
                    with Timeout(seconds=30, error_message=f"env.step() on step {step} timed out"):
                        obs, shaped_reward, done, info = self.env.step(action)
                    
                    total_shaped_reward += shaped_reward; step += 1
                except TimeoutError as e:
                    print(f"\n[WARN] Trajectory failed: {e}. Returning reward so far.", flush=True)
                    break # Exit loop and return reward so far
                except Exception as e:
                    print(f"\n[WARN] Trajectory step failed: {e}. Returning reward so far.", flush=True)
                    break # Exit loop
            return total_shaped_reward

    # ========================================================================
    # HELPER CLASS: Stage2OnlineTrainer
    # (MODIFIED with timeouts)
    # ========================================================================

    class Stage2OnlineTrainer:
        def __init__(self, env, policy, value_cache, learning_rate: float, beta: float, batch_size: int, max_grad_norm: float, max_steps: int, log_dir: str, checkpoint_dir: str):
            self.env = env; self.policy = policy; self.value_cache = value_cache; self.beta = beta
            self.batch_size = batch_size; self.max_grad_norm = max_grad_norm; self.max_steps = max_steps
            self.optimizer = optim.AdamW(self.policy.parameters(), lr=learning_rate)
            self.log_dir = Path(log_dir); self.log_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.checkpoint_dir = Path(checkpoint_dir); self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
            self.global_step = 0; self.best_reward = -float('inf')
        
        def train(self, num_iterations: int, eval_frequency: int, save_frequency: int, temperature: float):
            print(f"\n{'='*60}\nStage 2: Online Policy Optimization\n{'='*60}")
            print(f"Iterations: {num_iterations}, Batch Size: {self.batch_size}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}, A-PO Scale (beta): {self.beta}\n{'='*60}\n")
            
            batch_data = []
            for iteration in range(num_iterations):
                print(f"-- Iteration {iteration + 1}/{num_iterations} --", flush=True)
                
                # --- NEW: Added timeout wrapper ---
                try:
                    trajectory_data = self._collect_trajectory(temperature)
                except TimeoutError as e:
                    print(f"\n[WARN] Trajectory collection failed: {e}. Skipping iteration.", flush=True)
                    continue
                except Exception as e:
                    print(f"\n[WARN] Trajectory collection failed: {e}. Skipping iteration.", flush=True)
                    continue

                batch_data.append(trajectory_data)
                print(f"  Rollout: R_shaped={trajectory_data['shaped_reward']:.3f}, R_raw={trajectory_data['raw_reward']:.3f}, V*={trajectory_data['v_star']:.3f}, Adv={trajectory_data['advantage']:.3f}", flush=True)
                
                if len(batch_data) >= self.batch_size:
                    # Count positive vs negative advantages
                    pos_advs = sum(1 for t in batch_data if t['advantage'] > 0)
                    neg_advs = len(batch_data) - pos_advs

                    metrics = self._update_policy(batch_data)
                    print(f"  Training (Batch Update):", flush=True)
                    print(f"    Loss: {metrics['loss']:.4f}, Mean Adv: {metrics['mean_advantage']:.4f}", flush=True)
                    print(f"    Advs: {pos_advs} positive, {neg_advs} negative", flush=True)

                    # Warning if all advantages are negative
                    if pos_advs == 0:
                        print(f"    ⚠️  All advantages negative - no positive signal!", flush=True)

                    self._log_metrics(metrics, iteration); batch_data = []; self.global_step += 1
                    # Clear GPU cache to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                
                if (iteration + 1) % eval_frequency == 0:
                    eval_stats = self._evaluate(num_episodes=10, temperature=0.7)
                    print(f"\n  Evaluation:", flush=True)
                    print(f"    Mean Shaped Reward: {eval_stats['mean_shaped_reward']:.3f}, Success: {eval_stats['success_rate']:.1%}", flush=True)
                    self._log_metrics(eval_stats, iteration, prefix="eval")
                    if eval_stats['mean_shaped_reward'] > self.best_reward:
                        self.best_reward = eval_stats['mean_shaped_reward']
                        self._save_checkpoint("best_model"); print(f"    ✓ New best model!", flush=True)
                
                if (iteration + 1) % save_frequency == 0:
                    self._save_checkpoint(f"checkpoint_{iteration+1}"); print(f"  ✓ Checkpoint saved", flush=True)
            
            print(f"\n{'='*60}\nTraining Complete!\nBest Mean Shaped Reward: {self.best_reward:.3f}\n{'='*60}\n", flush=True)
            self.writer.close()
        
        def _collect_trajectory(self, temperature: float) -> Dict:
            # --- NEW: Added 60 second timeout to reset ---
            with Timeout(seconds=60, error_message="env.reset() timed out"):
                obs, info = self.env.reset()
            
            instruction = info.get("instruction", ""); v_star = self.value_cache.get(instruction, default=0.0)
            actions, observations = [], [obs]; done, step = False, 0
            total_shaped_reward = 0.0; final_step_info = {}
            
            while not done and step < self.max_steps:
                action, _ = self.policy.generate_action(
                    instruction=instruction, observation=obs, temperature=temperature, sample=True
                )
                actions.append(action)
                
                # --- NEW: Added 30 second timeout to step ---
                with Timeout(seconds=30, error_message=f"env.step() on step {step} timed out"):
                    obs, shaped_reward, done, step_info = self.env.step(action)
                
                observations.append(obs); final_step_info = step_info
                total_shaped_reward += shaped_reward; step += 1
            
            raw_reward = final_step_info.get("raw_reward", 0.0)
            advantage = total_shaped_reward - v_star
            return {"instruction": instruction, "observations": observations[:-1], "actions": actions, "raw_reward": raw_reward, "shaped_reward": total_shaped_reward, "v_star": v_star, "advantage": advantage, "num_steps": len(actions)}
        
        def _update_policy(self, batch_data: List[Dict]) -> Dict:
            """
            A*-PO Policy Gradient Loss (FIXED VERSION with Gradient Accumulation)
            Loss = -E[log π(y|x) * A*(x,y)]

            Uses gradient accumulation to avoid OOM on large batches.
            """
            # Collect advantages for normalization
            batch_advantages = [traj["advantage"] for traj in batch_data]
            advantages_tensor = torch.tensor(batch_advantages, dtype=torch.float32)

            # Normalize advantages
            if len(advantages_tensor) > 1:
                advantages_normalized = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            else:
                advantages_normalized = advantages_tensor

            # Clip advantages
            advantages_clipped = torch.clamp(advantages_normalized, -5.0, 5.0)

            # Gradient accumulation: process one trajectory at a time
            self.optimizer.zero_grad()
            accumulated_loss = 0.0
            log_probs_list = []

            for idx, traj in enumerate(batch_data):
                # Compute log prob for this trajectory
                total_log_prob = 0.0
                for obs, action in zip(traj["observations"], traj["actions"]):
                    log_prob, _ = self.policy.compute_log_probs(
                        instruction=traj["instruction"], observation=obs,
                        action=action, requires_grad=True
                    )
                    total_log_prob = total_log_prob + log_prob

                log_probs_list.append(total_log_prob.item())

                # Compute loss for this trajectory
                adv = advantages_clipped[idx]
                traj_loss = -(total_log_prob * adv) * self.beta / len(batch_data)

                # Backward pass (accumulates gradients)
                traj_loss.backward()
                accumulated_loss += traj_loss.item()

                # Clear GPU cache after each trajectory
                del total_log_prob, traj_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Clip gradients and update
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            return {
                "loss": accumulated_loss,
                "mean_log_prob": np.mean(log_probs_list),
                "mean_advantage": advantages_tensor.mean().item(),
                "std_advantage": advantages_tensor.std().item() if len(advantages_tensor) > 1 else 0.0
            }
        
        def _evaluate(self, num_episodes: int, temperature: float) -> Dict:
            shaped_rewards, raw_rewards, successes = [], [], []
            for _ in range(num_episodes):
                try:
                    traj = self._collect_trajectory(temperature)
                    shaped_rewards.append(traj["shaped_reward"]); raw_rewards.append(traj["raw_reward"])
                    successes.append(traj["raw_reward"] > 0.5)
                except (TimeoutError, Exception) as e:
                    print(f"\n[WARN] Evaluation trajectory failed: {e}. Skipping.", flush=True)
                    continue
            if not shaped_rewards: # All eval episodes failed
                return {"mean_shaped_reward": -99.0, "mean_raw_reward": -99.0, "std_raw_reward": 0.0, "success_rate": 0.0}
            return {"mean_shaped_reward": np.mean(shaped_rewards), "mean_raw_reward": np.mean(raw_rewards), "std_raw_reward": np.std(raw_rewards), "success_rate": np.mean(successes)}
        
        def _log_metrics(self, metrics: Dict, iteration: int, prefix: str = "train"):
            for key, value in metrics.items(): self.writer.add_scalar(f"{prefix}/{key}", value, iteration)
        
        def _save_checkpoint(self, name: str):
            save_path = self.checkpoint_dir / name; save_path.mkdir(exist_ok=True, parents=True)
            self.policy.save(str(save_path / "policy"))
            state = {"global_step": self.global_step, "best_reward": self.best_reward}
            with open(save_path / "state.json", "w") as f: json.dump(state, f)

    # ========================================================================
    # MAIN EXECUTION SCRIPT
    # (This logic is unchanged, but calls the new robust classes)
    # ========================================================================

    print("\n" + "=" * 60, flush=True)
    print("A*-PO Modular Training on H100 (Single File, Robust Timeouts)", flush=True)
    print("=" * 60, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Device: {device}", flush=True)
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    NUM_PRODUCTS = config.get("num_products", 1000); USE_HUMAN_GOALS = True
    index_path = "/data/webshop_indexes"; docs_path = "/data/webshop_resources/documents.jsonl"

    if not os.path.exists(index_path):
        print("\n📦 Building search indexes...", flush=True)
        os.makedirs("/data/webshop_resources", exist_ok=True)
        from web_agent_site.engine.engine import load_products
        data_files = ["/opt/WebShop/data/items_shuffle_1000.json", "/opt/WebShop/data/items_ins_v2_1000.json", "/opt/WebShop/data/items_human_ins.json"]
        all_products = []
        for f_path in data_files:
            if os.path.exists(f_path):
                products, *_ = load_products(filepath=f_path, num_products=NUM_PRODUCTS, human_goals=USE_HUMAN_GOALS); all_products.extend(products)
        seen_asins = set(); unique_products = []
        for p in all_products:
            asin = p.get("asin");
            if asin and asin not in seen_asins: unique_products.append(p); seen_asins.add(asin)
        print(f"Total unique products for index: {len(unique_products)}", flush=True)
        with open(docs_path, "w") as f:
            for p in unique_products:
                options_text = [f"{k}: {v if not isinstance(v, list) else ', '.join(v)}" for k, v in p.get("options", {}).items()]
                contents = " ".join([str(p.get("Title", "")), str(p.get("Description", "")), " ".join(p.get("BulletPoints", []) or []), ", ".join(options_text)]).lower()
                doc = {"id": p.get("asin", str(p.get("product_id", ""))), "contents": contents}; f.write(json.dumps(doc) + "\n")
        subprocess.run(
            [sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection", "--input", "/data/webshop_resources",
             "--index", index_path, "--generator", "DefaultLuceneDocumentGenerator", "--threads", "2", "--storePositions", "--storeDocvectors", "--storeRaw"], check=True
        )
        print("✓ Indexes built\n", flush=True); volume.commit()
    else: print("\n✓ Using cached indexes\n", flush=True)

    os.makedirs("/opt/WebShop/search_engine", exist_ok=True)
    indexes_name = "indexes_1k" if NUM_PRODUCTS >= 1000 else "indexes_100"
    symlink_path = f"/opt/WebShop/search_engine/{indexes_name}"
    if not os.path.exists(symlink_path): os.symlink(index_path, symlink_path)

    log_dir = f"/data/{config.get('log_dir', 'logs')}"; checkpoint_dir = f"/data/{config.get('checkpoint_dir', 'checkpoints')}"
    env = WebShopEnvironment(num_products=NUM_PRODUCTS, max_steps=config.get('max_steps', 20), human_goals=USE_HUMAN_GOALS)
    policy = RAGENPolicy(model_name=config.get('model_name'), max_length=config.get('max_length', 512), device=device)
    value_cache = OptimalValueCache(cache_file=f"{checkpoint_dir}/v_star_cache.json")

    if not config.get('skip_stage1', False):
        stage1_trainer = Stage1OfflineTrainer(
            env=env, policy=policy, value_cache=value_cache, k_samples=config.get('k_samples', 3),
            max_steps=config.get('max_steps', 20), temperature=config.get('temperature', 1.0)
        )
        stage1_trainer.collect_optimal_values(num_instructions=config.get('num_instructions_stage1', 10)); volume.commit()
    else: print("\nSkipping Stage 1 (using existing V* cache)")

    stage2_trainer = Stage2OnlineTrainer(
        env=env, policy=policy, value_cache=value_cache, learning_rate=config.get('learning_rate', 5e-6),
        beta=config.get('beta', 0.5), batch_size=config.get('batch_size', 4),
        max_grad_norm=config.get('max_grad_norm', 1.0), max_steps=config.get('max_steps', 20),
        log_dir=log_dir, checkpoint_dir=checkpoint_dir
    )
    stage2_trainer.train(
        num_iterations=config.get('num_iterations', 100), eval_frequency=config.get('eval_frequency', 10),
        save_frequency=config.get('save_frequency', 25), temperature=config.get('temperature', 0.7)
    )
    volume.commit()
    print("\n✓ A*-PO training completed successfully on Modal!", flush=True)
    return {"status": "success", "best_reward": stage2_trainer.best_reward, "v_star_stats": value_cache.get_statistics(), "log_dir": log_dir, "checkpoint_dir": checkpoint_dir}

# --------------------------------------------------------------------------------
# Local Entrypoint
# (This is unchanged from the last working version)
# --------------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    # Environment
    num_products: int = 1000, max_steps: int = 20,
    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_length: int = 512,
    # Stage 1: Offline
    skip_stage1: bool = False, num_instructions_stage1: int = 50, k_samples: int = 4,
    # Stage 2: Online
    num_iterations: int = 500, learning_rate: float = 5e-6, beta: float = 0.5,
    batch_size: int = 4, max_grad_norm: float = 1.0, temperature: float = 0.8,
    # Logging
    log_dir: str = "logs", checkpoint_dir: str = "checkpoints",
    eval_frequency: int = 25, save_frequency: int = 50
):
    import json
    config = {
        "num_products": num_products, "max_steps": max_steps, "model_name": model_name,
        "max_length": max_length, "skip_stage1": skip_stage1,
        "num_instructions_stage1": num_instructions_stage1, "k_samples": k_samples,
        "num_iterations": num_iterations, "learning_rate": learning_rate, "beta": beta,
        "batch_size": batch_size, "max_grad_norm": max_grad_norm, "temperature": temperature,
        "log_dir": log_dir, "checkpoint_dir": checkpoint_dir, "eval_frequency": eval_frequency,
        "save_frequency": save_frequency,
    }
    print("Starting A*-PO training on Modal (Robust Version)...", flush=True)
    print("Config:", json.dumps(config, indent=2))
    result = run_training_on_modal.remote(config)
    print(f"\n{'=' * 60}", flush=True)
    print("Final Results:", flush=True)
    print(json.dumps(result, indent=2))
    print(f"{'=' * 60}", flush=True)