from __future__ import annotations

import argparse
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch
import torch.optim as optim

from .data_loader import (
    EquationDataset,
    available_equation_ids,
    load_equation_datasets,
    split_equation_ids,
)
from .env import SymbolicRegressionEnv
from .grammar import Grammar
from .policy import MLPPolicy
from .reward import evaluate_expression_reward



def compute_returns(rewards: Sequence[float], gamma: float) -> torch.Tensor:
    # Compute discounted returns and normalise them for stable gradient updates
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    if len(returns_tensor) > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
    return returns_tensor


def evaluate_on_split(
    expression: str,
    equation: EquationDataset,
    sample_size: int = 1000,
    invalid_reward: float = -100.0,
) -> float:
    inputs, targets = equation.sample_uniform(
        sample_size=sample_size,
    )

    return evaluate_expression_reward(
        expression,
        inputs,
        targets,
        invalid_reward=invalid_reward,
    )


def evaluate_policy_on_equations(
    policy: MLPPolicy,
    equations: Sequence[EquationDataset],
    max_depth: int,
    max_steps: int,
    sample_size: int,
    episodes_per_equation: int = 5,
) -> tuple[float, dict[str, float]]:
    if not equations:
        return float("nan"), {}

    grammar = Grammar(num_variables=9)
    per_equation_scores: dict[str, float] = {}
    all_rewards: list[float] = []

    policy.eval()
    with torch.no_grad():
        for equation in equations:
            eq_id = equation.equation_id or equation.formula_name
            eval_env = SymbolicRegressionEnv(
                grammar=grammar,
                equations=[equation],
                max_depth=max_depth,
                max_steps=max_steps,
                sample_size=sample_size,
                num_variables=9,
            )

            rewards = []
            for _ in range(episodes_per_equation):
                state, _ = eval_env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action_mask = torch.tensor(
                        state[
                            eval_env.grammar.num_symbols + 1 : eval_env.grammar.num_symbols + 1 + eval_env.num_actions
                        ],
                        dtype=torch.float32,
                    ).unsqueeze(0)

                    probs = policy(state_tensor, action_mask)
                    action = int(torch.argmax(probs, dim=-1).item())

                    next_state, reward, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state

                rewards.append(float(episode_reward))

            eq_mean = float(np.mean(rewards))
            per_equation_scores[eq_id] = eq_mean
            all_rewards.extend(rewards)

    policy.train()
    return float(np.mean(all_rewards)), per_equation_scores



def train(
    env: SymbolicRegressionEnv,
    policy: MLPPolicy,
    val_equations: Sequence[EquationDataset],
    test_equations: Sequence[EquationDataset],
    episodes: int = 5000,
    gamma: float = 0.99,
    lr: float = 1e-4,
    debug_episode_log: bool = False,
    eval_every: int = 1000,
    checkpoint_dir: str = "checkpoints",
    entropy_coef: float = 0.01,
    batch_episodes: int = 20,
) -> None:
    if batch_episodes < 1:
        raise ValueError("batch_episodes must be >= 1")

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_dir = checkpoint_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    best_val_mean = float("-inf")

    # Track the best expression and reward found per equation
    best_by_equation: dict[str, dict[str, float | str | None]] = {}
    for equation in env.equations:
        eq_id = equation.equation_id or equation.formula_name
        best_by_equation[eq_id] = {
            "best_train": float("-inf"),
            "best_resample": float("-inf"),
            "expression": None,
        }
    
    training_logs = []
    recent_rewards: list[float] = []
    batch_losses: list[torch.Tensor] = []
    batch_rewards: list[float] = []

    for episode in range(episodes):
        episode_t0 = time.perf_counter()
        state, _ = env.reset()
        log_probs = []
        rewards = []
        entropies = []
        done = False
        final_info = {}
        last_step_seconds = 0.0

        
        while not done:
            
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_mask = torch.tensor(
                state[
                    env.grammar.num_symbols + 1 : env.grammar.num_symbols + 1 + env.num_actions
                ],
                dtype=torch.float32,
            ).unsqueeze(0)

            probs = policy(state_tensor, action_mask)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            entropies.append(dist.entropy())

        
            step_t0 = time.perf_counter()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            last_step_seconds = time.perf_counter() - step_t0
            done = terminated or truncated

           
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            final_info = info

      
        returns = compute_returns(rewards, gamma)
        # REINFORCE loss: scale log-probs by discounted returns, subtract entropy bonus
        policy_loss = torch.stack(
            [-log_prob * ret for log_prob, ret in zip(log_probs, returns)]
        ).sum()
        entropy_bonus = torch.stack(entropies).sum()
        episode_loss = policy_loss - entropy_coef * entropy_bonus

        batch_losses.append(episode_loss)

        episode_reward = float(sum(rewards))
        batch_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        current_expression = final_info.get("expression")
        current_eq = env.current_equation
        eq_id = (current_eq.equation_id or current_eq.formula_name) if current_eq else "unknown"

        # Update per-equation best results
        if eq_id in best_by_equation:
            if episode_reward > best_by_equation[eq_id]["best_train"]:
                best_by_equation[eq_id]["best_train"] = episode_reward
                best_by_equation[eq_id]["expression"] = current_expression

                if current_expression is not None:
                    best_by_equation[eq_id]["best_resample"] = evaluate_on_split(
                        current_expression,
                        current_eq,
                        sample_size=env.sample_size,
                        invalid_reward=env.invalid_reward,
                    )

        training_logs.append(
            {
                "episode": episode,
                "equation": eq_id,
                "train_reward": episode_reward,
                "best_train_reward": best_by_equation[eq_id]["best_train"] if eq_id in best_by_equation else float("-inf"),
                "best_resample_reward": best_by_equation[eq_id]["best_resample"] if eq_id in best_by_equation else float("-inf"),
                "expression": final_info.get("expression", ""),
            }
        )

        if debug_episode_log:
            equation_id = ""
            if env.current_equation is not None:
                equation_id = env.current_equation.equation_id or env.current_equation.formula_name
            print(f"Current equation: {equation_id}")
            print(f"Allowed vars: {sorted(env.current_allowed_vars)}")
            print(f"Generated expr: {final_info.get('expression', '')}")
            print(f"Expression length: {final_info.get('expression_length', 0)}")
            print(f"Last step time (s): {last_step_seconds:.6f}")
            print(f"Reward eval time (s): {float(final_info.get('reward_eval_seconds', 0.0)):.6f}")

        if (episode + 1) % 10 == 0:
            best_train = best_by_equation[eq_id]["best_train"] if eq_id in best_by_equation else float("-inf")
            best_resample = best_by_equation[eq_id]["best_resample"] if eq_id in best_by_equation else float("-inf")
            avg_reward_10 = float(np.mean(recent_rewards))
            episode_seconds = time.perf_counter() - episode_t0
            print(
                f"Episode {episode:04d} | "
                f"Equation: {eq_id} | "
                f"Allowed vars: {sorted(env.current_allowed_vars)} | "
                f"Train Reward: {episode_reward:.6f} | "
                f"Avg Reward 10: {avg_reward_10:.6f} | "
                f"Best Train: {best_train:.6f} | "
                f"Best Resample: {best_resample:.6f} | "
                f"Expr Len: {final_info.get('expression_length', 0)} | "
                f"Reward Time: {float(final_info.get('reward_eval_seconds', 0.0)):.4f}s | "
                f"Episode Time: {episode_seconds:.4f}s | "
                f"Expr: {final_info.get('expression', '')}"
            )

        # Update weights once we have collected enough episodes in the batch
        if (episode + 1) % batch_episodes == 0:
            loss = torch.stack(batch_losses).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.clear()
            batch_rewards.clear()

        # Periodically evaluate on validation set and save if improved
        if (episode + 1) % eval_every == 0:
            val_mean, val_scores = evaluate_policy_on_equations(
                policy,
                val_equations,
                max_depth=env.max_depth,
                max_steps=env.max_steps,
                sample_size=env.sample_size,
            )

            print(f"\n[Validation @ Episode {episode + 1}] Mean reward: {val_mean:.6f}")

            if not np.isnan(val_mean) and val_mean > best_val_mean:
                best_val_mean = val_mean
                torch.save(
                    {
                        "episode": episode + 1,
                        "model_state_dict": policy.state_dict(),
                        "best_val_mean": best_val_mean,
                        "val_scores": val_scores,
                    },
                    checkpoint_path / "best_policy.pt",
                )
                print(
                    f"Saved new best checkpoint at episode {episode + 1} "
                    f"with val reward {best_val_mean:.6f}\n"
                )

    # Flush remaining episodes that didn't fill a complete batch
    if batch_losses:
        loss = torch.stack(batch_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        batch_losses.clear()
        batch_rewards.clear()

    # Compute resampled rewards for best expressions on training equations
    import csv

    with open(log_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "equation",
                "train_reward",
                "best_train_reward",
                "best_resample_reward",
                "expression",
            ],
        )
        writer.writeheader()
        writer.writerows(training_logs)

    with open(log_dir / "best_expressions.txt", "w") as f:
        f.write("Per-equation best results:\n\n")
        for eq_id in sorted(best_by_equation.keys()):
            info = best_by_equation[eq_id]
            expr = info["expression"]
            best_train = info["best_train"]
            best_resample = info["best_resample"]
            
            resampled_reward = env.invalid_reward
            if expr is not None:
                # Evaluate on the same equation data using uniform resampling.
                for equation in env.equations:
                    if (equation.equation_id or equation.formula_name) == eq_id:
                        resampled_reward = evaluate_on_split(
                            expr,
                            equation,
                            sample_size=env.sample_size,
                            invalid_reward=env.invalid_reward,
                        )
                        break
            
            f.write(f"\n{eq_id}:\n")
            f.write(f"  Best train reward: {best_train:.6f}\n")
            f.write(f"  Best resampled reward: {best_resample:.6f}\n")
            f.write(f"  Resampled reward: {resampled_reward:.6f}\n")
            f.write(f"  Expression: {expr}\n")

    print("\nTraining finished")
    print("\nPer-equation results:")
    for eq_id in sorted(best_by_equation.keys()):
        info = best_by_equation[eq_id]
        print(f"  {eq_id}: best_train={info['best_train']:.6f}, best_resample={info['best_resample']:.6f}, expr={info['expression']}")

    val_mean, val_scores = evaluate_policy_on_equations(
        policy,
        val_equations,
        max_depth=env.max_depth,
        max_steps=env.max_steps,
        sample_size=env.sample_size,
    )
    test_mean, test_scores = evaluate_policy_on_equations(
        policy,
        test_equations,
        max_depth=env.max_depth,
        max_steps=env.max_steps,
        sample_size=env.sample_size,
    )

    print("\nHeld-out equation generalization:")
    if val_scores:
        print(f"  Validation mean reward: {val_mean:.6f}")
        for eq_id in sorted(val_scores):
            print(f"    {eq_id}: {val_scores[eq_id]:.6f}")
    else:
        print("  Validation mean reward: n/a (no validation equations)")

    if test_scores:
        print(f"  Test mean reward: {test_mean:.6f}")
        for eq_id in sorted(test_scores):
            print(f"    {eq_id}: {test_scores[eq_id]:.6f}")
    else:
        print("  Test mean reward: n/a (no test equations)")



def build_env(equation_ids: Sequence[str], max_depth: int, max_steps: int, sample_size: int):
    equations = load_equation_datasets(equation_ids)
    grammar = Grammar(num_variables=9)
    return SymbolicRegressionEnv(
        grammar=grammar,
        equations=equations,
        max_depth=max_depth,
        max_steps=max_steps,
        sample_size=sample_size,
        num_variables=9,
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP REINFORCE policy for symbolic regression.")
    parser.add_argument("--equation", dest="equations", action="append", help="Equation ID to load. Repeat to rotate between same-arity equations.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-episode-log", action="store_true")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--batch-episodes", type=int, default=20)
    args = parser.parse_args()


    # Fix random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.equations:
        all_equation_ids = args.equations
    else:
        all_equation_ids = available_equation_ids()

    train_ids, val_ids, test_ids = split_equation_ids(
        all_equation_ids,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=args.seed,
    )

    print(f"Total equations: {len(all_equation_ids)}")
    print(f"Train equations: {len(train_ids)}")
    print(f"Validation equations: {len(val_ids)}")
    print(f"Test equations: {len(test_ids)}")

    if not train_ids:
        raise ValueError("No training equations selected after split. Provide more equation IDs.")

    env = build_env(train_ids, args.max_depth, args.max_steps, args.sample_size)
    val_equations = load_equation_datasets(val_ids)
    test_equations = load_equation_datasets(test_ids)
    policy = MLPPolicy(env.state_dim, env.num_actions, hidden_dim=args.hidden_dim)

    print(f"Training on equations: {train_ids}")
    print(f"State dim: {env.state_dim} | Actions: {env.num_actions}")
    train(
        env,
        policy,
        val_equations=val_equations,
        test_equations=test_equations,
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        debug_episode_log=args.debug_episode_log,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        entropy_coef=args.entropy_coef,
        batch_episodes=args.batch_episodes,
    )


if __name__ == "__main__":
    main()