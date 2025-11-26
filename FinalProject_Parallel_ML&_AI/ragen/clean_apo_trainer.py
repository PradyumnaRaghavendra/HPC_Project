"""
Clean A*PO Trainer for WebShop

Key ideas:
- Clean, action-only system prompt.
- For each task, sample multiple trajectories from current policy.
- Use a "best-vs-rest" A*PO-style objective:
    - Higher return trajectories get positive advantage.
    - Lower return trajectories get negative advantage.
- Only optimize log-probs on ACTION tokens.
- KL regularization vs a frozen reference model.
- Designed to be stable and actually improve success rate.

This is A*PO-flavored, preference-style RL (practical and trainable),
not the broken Q - V* implementation.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re


# ===================== CONFIG =====================

@dataclass
class APOConfig:
    # Number of trajectories to sample per task per update
    trajectories_per_task: int = 2

    # Max environment steps per trajectory
    max_steps: int = 5

    # Optimizer / updates
    learning_rate: float = 5e-6
    clip_grad_norm: float = 1.0

    # Advantage handling
    adv_clip: float = 5.0

    # KL regularization vs ref model
    beta_kl: float = 0.05   # weight on D_KL(π || π_ref)

    # Sampling
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 24  # short, to force clean actions


# ===================== TRAINER =====================

class CleanAPOTrainer:
    """
    Practical A*PO-style trainer for WebShop with clean prompts.

    High-level:
      For each task:
        - Sample K trajectories from current policy.
        - Compute total return for each trajectory.
        - Compute per-trajectory advantages:
              A_j = R_j - mean(R_1..K)
        - For each action token in each trajectory:
              L += A_j * NLL(action_token)
           (plus KL penalty vs reference)
        - Positive A_j → decrease NLL (increase prob)
          Negative A_j → increase NLL (decrease prob)

    This is:
      - On-policy, candidate-based, advantage-weighted logprob optimization.
      - Strongly related to preference / A*PO style objectives.
    """

    def __init__(self, model, tokenizer, ref_model, env, config: APOConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.env = env
        self.config = config

        # Freeze ref model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Precompile simple action formats
        self.valid_action_regexes = [
            re.compile(r"^search\[[^\[\]]+\]$"),
            re.compile(r"^click\[B[0-9A-Z]+\]$"),
            re.compile(r"^buy now$"),
            re.compile(r"^back$"),
        ]

    # ===================== PROMPTS & ACTIONS =====================

    def get_system_prompt(self) -> str:
        return (
            "You are a WebShop agent. Output ONLY actions.\n\n"
            "VALID ACTIONS:\n"
            "search[query]\n"
            "click[B0XXXXXXX]\n"
            "buy now\n"
            "back\n\n"
            "RULES:\n"
            "1. Output ONLY the action, nothing else.\n"
            "2. No explanations, no reasoning, no natural language.\n"
            "3. One action per turn.\n"
        )

    def extract_action(self, raw: str) -> str:
        """
        Extract a valid action from model output.
        More permissive than strict regex to help learning,
        but returns canonical forms.
        """
        raw = raw.strip()
        low = raw.lower()

        # search[...]
        m = re.search(r"search\[([^\]]+)\]", raw, re.IGNORECASE)
        if m:
            q = m.group(1)
            q = q.replace("%20", " ").replace("%2C", " ")
            q = " ".join(q.split())
            q = " ".join(q.split()[:6])
            if len(q) > 2:
                return f"search[{q}]"

        # search: ...
        m = re.search(r"search[: ]+(.+)", low)
        if m:
            q = " ".join(m.group(1).split()[:6])
            if len(q) > 2:
                return f"search[{q}]"

        # click[B0...]
        m = re.search(r"(B0[0-9A-Z]{8})", raw)
        if m:
            return f"click[{m.group(1)}]"

        # buy now / buy
        if "buy now" in low:
            return "buy now"
        if low.startswith("buy") or " buy " in low:
            return "buy now"

        # back
        if "back" in low:
            return "back"

        return ""  # invalid / no action found

    def is_valid_action(self, action: str) -> bool:
        return any(r.match(action) for r in self.valid_action_regexes)

    def format_penalty(self, action: str) -> float:
        """
        Reward shaping for syntax / usefulness:
          - strong penalty for empty/invalid
          - mild penalty for generic or junk
        """
        if not action:
            return -1.5
        if not self.is_valid_action(action):
            return -0.8
        # Very generic searches: slightly penalize
        generic_queries = {
            "search[products]",
            "search[item]",
            "search[stuff]",
            "search[buy now]",
        }
        if action.lower() in generic_queries:
            return -0.2
        return 0.0

    # ===================== TRAJECTORY SAMPLING =====================

    @torch.no_grad()
    def sample_action(self, prompt: str) -> str:
        """Sample a single action string from the model given a prompt."""
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

    def generate_trajectory(self, task: str) -> Dict:
        """
        Generate ONE trajectory for a task using the current policy.
        Stores:
          - (prompt, action) pairs per step
          - rewards
          - success flag
        """
        system_prompt = self.get_system_prompt()
        obs = self.env.reset(task)

        traj = {
            "task": task,
            "prompts": [],
            "actions": [],
            "rewards": [],
            "success": False,
        }

        for t in range(self.config.max_steps):
            user_prompt = f"Task: {task}\nState: {obs}\n\nAction:"
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            raw = self.sample_action(full_prompt)
            action = self.extract_action(raw)

            pen = self.format_penalty(action)

            # Hard fail on totally broken
            if pen <= -1.5:
                traj["prompts"].append(full_prompt)
                traj["actions"].append("[INVALID]")
                traj["rewards"].append(-1.5)
                break

            if not self.is_valid_action(action):
                # soft end: invalid but parseable-ish
                traj["prompts"].append(full_prompt)
                traj["actions"].append("[INVALID]")
                traj["rewards"].append(pen)
                break

            # Step env
            obs, env_r, done, info = self.env.step(action)

            # Reward shaping
            total_r = env_r + pen + 0.1  # bonus for syntactically clean action

            traj["prompts"].append(full_prompt)
            traj["actions"].append(action)
            traj["rewards"].append(total_r)

            if done:
                traj["success"] = info.get("success", False)
                break

        return traj

    # ===================== LOSS COMPUTATION =====================

    def _action_logprobs(
        self,
        prompt: str,
        action: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (log π, log π_ref) ONLY over the action tokens.
        Returns:
          log_prob (policy), log_prob_ref (ref), each scalar.
        """

        device = self.model.device

        # Tokenize prompt and action separately
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

        # If somehow empty action, just return zeros
        if action_ids.numel() == 0:
            zero = torch.zeros((), device=device)
            return zero, zero

        input_ids = torch.cat([prompt_ids, action_ids], dim=1)
        # Labels: only action tokens contribute
        labels = torch.full_like(input_ids, -100)
        labels[:, prompt_ids.shape[1]:] = input_ids[:, prompt_ids.shape[1]:]

        # Policy log prob (with grads)
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            use_cache=False,
        )
        # loss is mean NLL on action tokens
        nll = outputs.loss
        logprob = -nll  # average log prob per action token

        # Ref log prob (no grad)
        with torch.no_grad():
            ref_out = self.ref_model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
            )
            ref_nll = ref_out.loss
            ref_logprob = -ref_nll

        return logprob, ref_logprob

    def compute_loss(
        self,
        all_task_trajectories: List[List[Dict]],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        all_task_trajectories: list of length = num_tasks,
        each element is a list of trajectories sampled for that task.

        Implements:
            For each task:
              - compute returns R_j
              - baseline = mean_j R_j
              - advantage A_j = clip(R_j - baseline)
              - for each action in traj j:
                    L += A_j * NLL(action) + beta * KL_token
        """

        device = self.model.device
        total_loss = None
        all_advs = []
        all_kls = []
        used_steps = 0

        self.model.eval()
        self.ref_model.eval()

        for trajs in all_task_trajectories:
            if len(trajs) == 0:
                continue

            # Returns per trajectory
            returns = [sum(tr["rewards"]) for tr in trajs]
            if all(r == 0.0 for r in returns):
                continue

            returns_t = torch.tensor(returns, device=device, dtype=torch.float32)
            baseline = returns_t.mean().item()  # simple baseline
            adjs = returns_t - baseline

            # Normalize & clip advantages per task
            if adjs.numel() > 1 and adjs.std() > 1e-6:
                adjs = (adjs - adjs.mean()) / (adjs.std() + 1e-8)
            adjs = torch.clamp(adjs, -self.config.adv_clip, self.config.adv_clip)

            for j, traj in enumerate(trajs):
                A_j = adjs[j].item()
                if abs(A_j) < 1e-6:
                    continue  # no signal for this traj

                all_advs.append(A_j)

                for prompt, action in zip(traj["prompts"], traj["actions"]):
                    if action == "[INVALID]":
                        continue

                    logp, logp_ref = self._action_logprobs(prompt, action)
                    if (
                        torch.isnan(logp)
                        or torch.isinf(logp)
                        or torch.isnan(logp_ref)
                        or torch.isinf(logp_ref)
                    ):
                        continue

                    # KL approx per action
                    kl = (logp - logp_ref)  # log π - log π_ref
                    all_kls.append(kl.item())

                    # Policy gradient loss: -A * logπ
                    # Since logp is average per-token, this is fine.
                    # For A>0 → increase logp; A<0 → decrease logp.
                    pg_loss = -A_j * logp

                    # KL regularization
                    kl_loss = self.config.beta_kl * kl

                    step_loss = pg_loss + kl_loss

                    if (
                        torch.isnan(step_loss)
                        or torch.isinf(step_loss)
                    ):
                        continue

                    if total_loss is None:
                        total_loss = step_loss
                    else:
                        total_loss = total_loss + step_loss

                    used_steps += 1

        if total_loss is None or used_steps == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            metrics = {
                "mean_adv": 0.0,
                "mean_kl": 0.0,
                "used_steps": 0,
            }
            return loss, metrics

        loss = total_loss / used_steps
        mean_adv = sum(all_advs) / len(all_advs) if all_advs else 0.0
        mean_kl = sum(all_kls) / len(all_kls) if all_kls else 0.0

        metrics = {
            "mean_adv": float(mean_adv),
            "mean_kl": float(mean_kl),
            "used_steps": int(used_steps),
        }
        return loss, metrics

    # ===================== TRAIN STEP =====================

    def train_step(self, tasks: List[str], verbose: bool = True) -> Dict:
        """
        For a batch of tasks:
          1. Sample multiple trajectories per task.
          2. Compute A*PO-style loss.
          3. Backprop + optimizer step.
        """
        if verbose:
            print(f"\n🎯 Collecting trajectories for {len(tasks)} tasks...", flush=True)

        self.model.eval()

        all_task_trajs: List[List[Dict]] = []
        total_rewards = []
        total_successes = 0
        total_trajs = 0

        # 1) Sample trajectories
        for ti, task in enumerate(tasks):
            if verbose:
                print(f"\n  Task {ti+1}/{len(tasks)}: {task[:80]}...", flush=True)

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
                        f"steps={len(traj['actions'])}, R={R:.2f}, "
                        f"succ={traj['success']}",
                        flush=True,
                    )

            all_task_trajs.append(task_trajs)

        # 2) Compute loss
        loss, metrics = self.compute_loss(all_task_trajs)

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

        # 3) Update
        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.clip_grad_norm,
        )

        if (
            torch.isnan(grad_norm)
            or torch.isinf(grad_norm)
        ):
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
