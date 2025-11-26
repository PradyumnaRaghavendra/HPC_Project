"""
Train WebShop agent with Clean A*PO

Design:
- Clean, action-only prompting.
- For each task, sample multiple trajectories from current policy.
- Candidate-based A*PO-style update:
    For each task:
      - R_j = total return of trajectory j.
      - A_j = normalized( R_j - mean_j R ).
      - For each action in trajectory j:
            loss += -A_j * log π(a|s)  +  β * (log π(a|s) - log π_ref(a|s))
- Only compute log-probs on ACTION tokens (mask out prompt/system).
- Stable on H100: bf16, no use_cache in loss path, grad clipping, no OOM hacks.
- Less overzealous penalties, no skipping updates on "large but finite" gradients.
"""

import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import modal

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
        "mkdir -p /etc/apt/keyrings",
        "wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc",
        "echo 'deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb bookworm main' | tee /etc/apt/sources.list.d/adoptium.list",
        "apt-get update",
        "apt-get install -y temurin-21-jdk",
    )
    .pip_install(
        "numpy>=1.25.0",
        "torch>=2.2.0",
        "transformers>=4.42.0",
        "datasets>=2.14.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "accelerate>=0.25.0",
        "faiss-cpu>=1.7.0",
        "pyserini",
        "gym==0.24.0",
        "spacy>=3.6.0",
        "flask==2.1.2",
        "Werkzeug==2.0.3",
        "beautifulsoup4",
        "rank-bm25",
        "nltk",
        "cleantext",
        "requests",
        "scikit-learn",
        "pandas",
        "selenium",
        "gdown",
        "rich",
        "thefuzz",
        "python-Levenshtein",
    )
    .run_commands("python -m spacy download en_core_web_sm")
)

app = modal.App("clean-apo-webshop", image=image)
volume = modal.Volume.from_name("clean-apo-outputs", create_if_missing=True)


@app.function(gpu="H100", timeout=3600, volumes={"/outputs": volume})
def train_clean_apo():
    import subprocess
    import sys
    import json
    import re
    from dataclasses import dataclass
    from typing import List, Dict, Tuple

    import torch
    from torch import bfloat16
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 60, flush=True)
    print("🚀 CLEAN A*PO TRAINING FOR WEBSHOP", flush=True)
    print("=" * 60 + "\n", flush=True)

    # ==================== WEBSHOP SETUP ====================
    print("📥 Setting up WebShop...", flush=True)

    if not os.path.exists("/root/WebShop"):
        subprocess.run(
            ["git", "clone", "https://github.com/princeton-nlp/WebShop.git", "/root/WebShop"],
            check=True,
        )

    os.chdir("/root/WebShop")
    os.makedirs("data", exist_ok=True)

    print("📥 Downloading data...", flush=True)
    for file_id, path in [
        ("1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib", "data/items_shuffle_1000.json"),
        ("1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu", "data/items_ins_v2_1000.json"),
        ("14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O", "data/items_human_ins.json"),
    ]:
        if not os.path.exists(path):
            subprocess.run([sys.executable, "-m", "gdown", file_id, "-O", path], check=False)

    # Build index for 100-product subset (matches env num_products=100)
    print("📥 Building indexes...", flush=True)
    os.makedirs("search_engine", exist_ok=True)
    sys.path.insert(0, "/root/WebShop")

    from web_agent_site.utils import DEFAULT_FILE_PATH
    from web_agent_site.engine.engine import load_products

    all_products, *_ = load_products(
        filepath=DEFAULT_FILE_PATH,
        num_products=100,
        human_goals=True,
    )
    os.makedirs("search_engine/resources_100", exist_ok=True)

    docs = []
    for p in all_products:
        option_texts = []
        for name, contents in p.get("options", {}).items():
            option_texts.append(f"{name}: {', '.join(contents)}")
        doc = {
            "id": p.get("asin", ""),
            "contents": " ".join(
                [
                    p.get("Title", ""),
                    p.get("Description", ""),
                    (p.get("BulletPoints") or [""])[0] if p.get("BulletPoints") else "",
                    ", and ".join(option_texts),
                ]
            ).lower(),
        }
        docs.append(doc)

    with open("search_engine/resources_100/documents.jsonl", "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            "search_engine/resources_100",
            "--index",
            "search_engine/indexes_100",
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ],
        check=True,
        cwd="/root/WebShop",
        capture_output=True,
    )

    print("✓ WebShop ready!", flush=True)
    os.environ["PYTHONPATH"] = "/root/WebShop"

    # ==================== MODELS ====================
    print("\n📦 Loading models (bf16)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    print("✓ Models loaded", flush=True)

    # ==================== ENV & TASKS ====================
    print("\n🌐 Creating environment...", flush=True)
    from web_agent_site.envs import WebAgentTextEnv

    env = WebAgentTextEnv(observation_mode="text", num_products=100)

    with open("/root/WebShop/data/items_ins_v2_1000.json") as f:
        raw = json.load(f)
        tasks = list(raw.values())[:100]

    print(f"✓ {len(tasks)} tasks loaded", flush=True)

    # ==================== CLEAN A*PO TRAINER ====================

    @dataclass
    class APOConfig:
        trajectories_per_task: int = 4   # more diversity per task
        max_steps: int = 6
        learning_rate: float = 5e-6
        clip_grad_norm: float = 1.0
        adv_clip: float = 2.0
        beta_kl: float = 0.01
        temperature: float = 0.3
        top_p: float = 0.9
        max_new_tokens: int = 24

    class CleanAPOTrainer:
        """
        Candidate-based A*PO-style trainer.
        """

        def __init__(self, model, tokenizer, ref_model, env, config: APOConfig):
            self.model = model
            self.tokenizer = tokenizer
            self.ref_model = ref_model
            self.env = env
            self.config = config

            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )

            self.valid_action_regexes = [
                re.compile(r"^search\[[^\[\]]+\]$"),
                re.compile(r"^click\[B[0-9A-Z]+\]$"),
                re.compile(r"^buy now$"),
                re.compile(r"^back$"),
            ]

        # ---------- prompt helpers ----------

        def get_system_prompt(self) -> str:
            return (
                "You are a WebShop agent.\n"
                "You MUST output exactly one action each turn.\n"
                "VALID ACTIONS ONLY (no extra text):\n"
                "  search[query]\n"
                "  click[B0XXXXXXX]\n"
                "  buy now\n"
                "  back\n"
                "No explanations. No dialogue. Just the action.\n"
            )

        def task_to_text(self, task) -> str:
            if isinstance(task, dict):
                instr = task.get("instruction")
                if isinstance(instr, str) and instr.strip():
                    return instr.strip()
                attrs = task.get("attributes") or []
                if attrs:
                    return "Find a product with: " + ", ".join(attrs)
            return str(task)

        # ---------- action parsing ----------

        def extract_action(self, raw: str) -> str:
            raw = raw.strip()
            low = raw.lower()

            # explicit search[...]
            m = re.search(r"search\[([^\]]+)\]", raw, re.IGNORECASE)
            if m:
                q = m.group(1)
                q = q.replace("%20", " ").replace("%2c", " ")
                q = " ".join(q.split())
                q = " ".join(q.split()[:8])
                if len(q) > 2:
                    return f"search[{q}]"

            # click asin
            m = re.search(r"\b(B0[0-9A-Z]{8})\b", raw)
            if m:
                return f"click[{m.group(1)}]"

            # buy/back
            if "buy now" in low or low.startswith("buy "):
                return "buy now"
            if " back" in low or low.startswith("back"):
                return "back"

            return ""

        def is_valid_action(self, action: str) -> bool:
            return any(r.match(action) for r in self.valid_action_regexes)

        def format_penalty(self, action: str) -> float:
            """
            Gentle shaping:
            - invalid / empty: small negative
            - valid: zero penalty (let env reward drive learning)
            """
            if not action:
                return -0.2
            if not self.is_valid_action(action):
                return -0.2
            return 0.0

        # ---------- sampling ----------

        @torch.no_grad()
        def sample_action(self, prompt: str) -> str:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.model.device)

            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            gen = out[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            return text.strip()

        def generate_trajectory(self, task) -> Dict:
            """
            Rollout one trajectory for a given task.
            DOES NOT terminate on first invalid: invalid gets penalty,
            keeps same obs, continues up to max_steps.
            """
            sys_prompt = self.get_system_prompt()
            task_text = self.task_to_text(task)

            obs = self.env.reset(task)
            traj = {
                "prompts": [],
                "actions": [],
                "rewards": [],
                "success": False,
            }

            for t in range(self.config.max_steps):
                prompt = f"{sys_prompt}\nTask: {task_text}\nState: {obs}\n\nAction:"
                raw = self.sample_action(prompt)
                action = self.extract_action(raw)
                pen = self.format_penalty(action)

                if t == 0:
                    print(f"      model_raw: {raw!r} -> action: {action!r}", flush=True)

                if not action or not self.is_valid_action(action):
                    # invalid: apply penalty, keep state, continue
                    traj["prompts"].append(prompt)
                    traj["actions"].append("[INVALID]")
                    traj["rewards"].append(pen)
                    continue

                # Valid action: step env
                obs, env_r, done, info = self.env.step(action)
                total_r = env_r + pen  # pen is 0 for valid; env_r can be shaped

                traj["prompts"].append(prompt)
                traj["actions"].append(action)
                traj["rewards"].append(total_r)

                if done:
                    traj["success"] = info.get("success", False)
                    break

            return traj

        # ---------- log-probs on action tokens ----------

        def _action_logprobs(self, prompt: str, action: str) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute log π and log π_ref of `action` tokens given `prompt`.
            Loss only on the action tokens.
            """
            device = self.model.device

            if not action or action == "[INVALID]":
                z = torch.tensor(0.0, device=device)
                return z, z

            prompt_enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            )
            action_enc = self.tokenizer(
                action,
                return_tensors="pt",
                add_special_tokens=False,
            )

            prompt_ids = prompt_enc["input_ids"].to(device)
            action_ids = action_enc["input_ids"].to(device)

            if action_ids.numel() == 0:
                z = torch.tensor(0.0, device=device)
                return z, z

            input_ids = torch.cat([prompt_ids, action_ids], dim=1)

            labels = torch.full_like(input_ids, -100)
            labels[:, prompt_ids.shape[1]:] = input_ids[:, prompt_ids.shape[1]:]

            # Policy
            out = self.model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
            )
            logp = -out.loss

            # Ref
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=input_ids,
                    labels=labels,
                    use_cache=False,
                )
                ref_logp = -ref_out.loss

            return logp, ref_logp

        # ---------- loss: candidate-based A*PO ----------

        def compute_loss(self, all_task_trajs: List[List[Dict]]) -> Tuple[torch.Tensor, Dict]:
            device = self.model.device
            self.model.eval()
            self.ref_model.eval()

            total_loss = None
            all_advs = []
            all_kls = []
            used_steps = 0

            for trajs in all_task_trajs:
                if not trajs:
                    continue

                returns = [sum(tr["rewards"]) for tr in trajs]
                if all(abs(r) < 1e-6 for r in returns):
                    # all zero-ish → no signal for this task
                    continue

                R = torch.tensor(returns, device=device, dtype=torch.float32)
                baseline = R.mean()
                advs = R - baseline

                if advs.numel() > 1 and advs.std() > 1e-6:
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                advs = torch.clamp(advs, -self.config.adv_clip, self.config.adv_clip)

                for j, traj in enumerate(trajs):
                    A_j = advs[j].item()
                    if abs(A_j) < 1e-6:
                        continue  # no preference signal

                    all_advs.append(A_j)

                    for prompt, action in zip(traj["prompts"], traj["actions"]):
                        if action == "[INVALID]":
                            continue

                        logp, logp_ref = self._action_logprobs(prompt, action)
                        if (
                            torch.isnan(logp) or torch.isinf(logp)
                            or torch.isnan(logp_ref) or torch.isinf(logp_ref)
                        ):
                            continue

                        kl = logp - logp_ref
                        all_kls.append(kl.item())

                        pg_loss = -A_j * logp
                        kl_loss = self.config.beta_kl * kl
                        step_loss = pg_loss + kl_loss

                        if torch.isnan(step_loss) or torch.isinf(step_loss):
                            continue

                        if total_loss is None:
                            total_loss = step_loss
                        else:
                            total_loss = total_loss + step_loss

                        used_steps += 1

            if total_loss is None or used_steps == 0:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                return loss, {
                    "mean_adv": 0.0,
                    "mean_kl": 0.0,
                    "used_steps": 0,
                }

            loss = total_loss / used_steps
            mean_adv = sum(all_advs) / len(all_advs) if all_advs else 0.0
            mean_kl = sum(all_kls) / len(all_kls) if all_kls else 0.0

            return loss, {
                "mean_adv": float(mean_adv),
                "mean_kl": float(mean_kl),
                "used_steps": int(used_steps),
            }

        # ---------- train step ----------

        def train_step(self, tasks: List[Dict], verbose: bool = True) -> Dict:
            if verbose:
                print(f"\n🎯 Collecting trajectories for {len(tasks)} tasks...", flush=True)

            self.model.eval()

            all_task_trajs: List[List[Dict]] = []
            total_rewards = []
            total_successes = 0
            total_trajs = 0

            for ti, task in enumerate(tasks):
                if verbose:
                    tt = self.task_to_text(task)
                    print(f"\n  Task {ti+1}/{len(tasks)}: {tt[:120]}...", flush=True)

                task_trajs = []
                for k in range(self.config.trajectories_per_task):
                    traj = self.generate_trajectory(task)
                    task_trajs.append(traj)

                    R = sum(traj["rewards"])
                    total_rewards.append(R)
                    total_trajs += 1
                    if traj["success"]:
                        total_successes += 1

                    if verbose:
                        tag = "✅" if traj["success"] else "➜"
                        print(
                            f"    {tag} traj {k+1}: "
                            f"steps={len(traj['actions'])}, "
                            f"R={R:.2f}, succ={traj['success']}, "
                            f"actions={traj['actions']}",
                            flush=True,
                        )

                all_task_trajs.append(task_trajs)

            loss, metrics = self.compute_loss(all_task_trajs)

            # safety: only skip on NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print("  ⚠️ NaN/Inf loss, skipping update", flush=True)
                avg_reward = sum(total_rewards) / max(1, len(total_rewards))
                success_rate = total_successes / max(1, total_trajs)
                return {
                    "loss": 0.0,
                    "reward": avg_reward,
                    "success_rate": success_rate,
                    **metrics,
                }

            # update
            self.model.train()
            self.optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_grad_norm,
            )

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if verbose:
                    print("  ⚠️ NaN/Inf grad, skipping step", flush=True)
                self.optimizer.zero_grad()
                avg_reward = sum(total_rewards) / max(1, len(total_rewards))
                success_rate = total_successes / max(1, total_trajs)
                return {
                    "loss": 0.0,
                    "reward": avg_reward,
                    "success_rate": success_rate,
                    **metrics,
                }

            # we DO NOT skip on large finite norm; it's clipped already
            self.optimizer.step()
            self.model.eval()
            torch.cuda.empty_cache()

            avg_reward = sum(total_rewards) / max(1, len(total_rewards))
            success_rate = total_successes / max(1, total_trajs)

            if verbose:
                print(
                    f"\n📊 METRICS:"
                    f"\n   Loss:        {loss.item():.4f}"
                    f"\n   Reward:      {avg_reward:.3f}"
                    f"\n   Success:     {success_rate:.1%}"
                    f"\n   Grad Norm:   {grad_norm.item():.3f}"
                    f"\n   Mean Adv:    {metrics['mean_adv']:.3f}"
                    f"\n   Mean KL:     {metrics['mean_kl']:.3f}"
                    f"\n   Used Steps:  {metrics['used_steps']}",
                    flush=True,
                )

            return {
                "loss": float(loss.item()),
                "reward": float(avg_reward),
                "success_rate": float(success_rate),
                "grad_norm": float(grad_norm.item()),
                **metrics,
            }

    # ==================== INIT TRAINER & LOOP ====================

    print("\n🎯 Creating A*PO trainer...", flush=True)
    config = APOConfig()
    trainer = CleanAPOTrainer(model, tokenizer, ref_model, env, config)
    print("✓ Trainer ready", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("🎬 STARTING CLEAN A*PO TRAINING", flush=True)
    print("=" * 60, flush=True)

    final_metrics = {}
    for step in range(50):
        print("\n" + "─" * 60, flush=True)
        print(f"📍 STEP {step + 1}/50", flush=True)
        print("─" * 60, flush=True)

        batch_start = (step * 4) % len(tasks)
        batch = tasks[batch_start: batch_start + 4]

        metrics = trainer.train_step(batch, verbose=True)
        final_metrics = metrics

        print("\n📊 STEP METRICS:", flush=True)
        print(f"   Loss:        {metrics['loss']:.4f}", flush=True)
        print(f"   Reward:      {metrics['reward']:.3f}", flush=True)
        print(f"   Success:     {metrics['success_rate']:.1%}", flush=True)
        print(f"   Grad Norm:   {metrics.get('grad_norm', 0):.3f}", flush=True)

        if (step + 1) % 10 == 0:
            print("\n💾 Saving checkpoint...", flush=True)
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "metrics": metrics,
                    "config": config.__dict__,
                },
                f"/outputs/clean_apo_step{step+1}.pt",
            )
            volume.commit()
            print(f"✓ Saved to /outputs/clean_apo_step{step+1}.pt", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("✅ TRAINING COMPLETE", flush=True)
    print("=" * 60, flush=True)

    torch.save(model.state_dict(), "/outputs/clean_apo_final.pt")
    volume.commit()

    return {"status": "complete", "final_metrics": final_metrics}


@app.local_entrypoint()
def main():
    print("🚀 Launching Clean A*PO training on Modal H100...")
    result = train_clean_apo.remote()
    print("\n✅ Training complete!")
    print(f"   Status: {result['status']}")
    print(f"   Final metrics: {result['final_metrics']}")
    print(
        "\n💾 Download results:\n"
        "   modal volume get clean-apo-outputs /outputs ./clean_apo_results"
    )
