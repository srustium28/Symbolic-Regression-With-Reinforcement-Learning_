from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from symbolic_rl.data_loader import (
    available_equation_ids,
    load_equation_datasets,
    split_equation_ids,
)
from symbolic_rl.env import SymbolicRegressionEnv
from symbolic_rl.grammar import Grammar
from symbolic_rl.policy import MLPPolicy
from symbolic_rl.reward import evaluate_expression_reward, safe_eval, simplify_expr


def recover_expression(policy: MLPPolicy, env: SymbolicRegressionEnv, device: torch.device, sample: bool) -> str:
    state, _ = env.reset()
    done = False
    expression = ""

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_mask = torch.tensor(
            state[
                env.grammar.num_symbols + 1 : env.grammar.num_symbols + 1 + env.num_actions
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            probs = policy(state_tensor, action_mask).squeeze(0)

        if sample:
            if torch.sum(probs) <= 0:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                action = int(np.random.choice(valid_actions))
            else:
                action = int(torch.multinomial(probs, 1).item())
        else:
            action = int(torch.argmax(probs).item())

        state, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        expression = info.get("expression", expression)

    return expression


def evaluate_expression(expression: str, equation, sample_size: int, invalid_reward: float) -> tuple[float, float]:
    inputs, targets = equation.sample_uniform(sample_size=sample_size)
    reward = evaluate_expression_reward(
        expression,
        inputs,
        targets,
        invalid_reward=invalid_reward,
    )

    if reward <= invalid_reward:
        return reward, float("inf")

    simplified_expression = simplify_expr(expression)

    try:
        predictions = safe_eval(simplified_expression, inputs)
    except Exception:
        return reward, float("inf")

    targets = np.asarray(targets, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)

    if predictions.shape[0] != targets.shape[0]:
        return reward, float("inf")
    if np.any(~np.isfinite(predictions)):
        return reward, float("inf")

    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    if not np.isfinite(rmse):
        return reward, float("inf")

    return reward, rmse


def evaluate_split(
    policy: MLPPolicy,
    equations,
    split_name: str,
    num_variables: int,
    max_depth: int,
    max_steps: int,
    sample_size: int,
    invalid_reward: float,
    n_rollouts: int,
    device: torch.device,
    out_file,
) -> float:
    rewards = []
    grammar = Grammar(num_variables=num_variables)

    out_file.write("=" * 60 + "\n")
    out_file.write(f"{split_name.upper()} EVALUATION\n")
    out_file.write(f"Rollouts per equation: {n_rollouts}\n")
    out_file.write("=" * 60 + "\n\n")

    for equation in equations:
        eq_id = equation.equation_id or equation.formula_name

        best_reward = float("-inf")
        best_rmse = float("inf")
        best_expr = ""

        env = SymbolicRegressionEnv(
            grammar=grammar,
            equations=[equation],
            max_depth=max_depth,
            max_steps=max_steps,
            sample_size=sample_size,
            invalid_reward=invalid_reward,
            num_variables=num_variables,
        )

        for _ in range(n_rollouts):
            expr = recover_expression(policy, env, device=device, sample=True)
            reward, rmse = evaluate_expression(
                expr,
                equation,
                sample_size=sample_size,
                invalid_reward=invalid_reward,
            )

            if reward > best_reward:
                best_reward = reward
                best_rmse = rmse
                best_expr = expr

        rewards.append(best_reward)

        out_file.write(f"{eq_id}:\n")
        out_file.write(f"  Best {split_name} reward: {best_reward:.6f}\n")
        out_file.write(f"  RMSE: {best_rmse:.6f}\n")
        out_file.write(f"  Expression: {best_expr}\n\n")

    mean_reward = float(np.mean(rewards)) if rewards else float("nan")
    out_file.write(f"Mean {split_name} reward: {mean_reward:.6f}\n\n")

    return mean_reward


def infer_policy_dims_from_checkpoint(checkpoint: dict) -> tuple[int, int, int]:
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = checkpoint

    first = state_dict.get("net.0.weight")
    out = state_dict.get("net.4.weight")
    if first is None or out is None:
        raise KeyError("Could not infer policy dimensions from checkpoint state_dict.")

    hidden_dim, state_dim = int(first.shape[0]), int(first.shape[1])
    num_actions = int(out.shape[0])
    return state_dim, num_actions, hidden_dim


def check_paths(run_dir: Path, checkpoint_path: Path, data_dir: Path, out_path: Path) -> None:
    print("Path check:")
    print(f"  run_dir_exists={run_dir.exists()} -> {run_dir}")
    print(f"  checkpoint_exists={checkpoint_path.exists()} -> {checkpoint_path}")
    print(f"  data_dir_exists={data_dir.exists()} -> {data_dir}")
    print(f"  output_parent_exists={out_path.parent.exists()} -> {out_path.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover best expressions on val/test splits from a saved checkpoint.")
    parser.add_argument("--run-dir", type=str, default="runs/batch1")
    parser.add_argument("--checkpoint-name", type=str, default="best_policy.pt")
    parser.add_argument("--data-dir", type=str, default="data/Feynman_with_units")
    parser.add_argument("--out-name", type=str, default="val_test_expressions.txt")
    parser.add_argument("--val-rollouts", type=int, default=50)
    parser.add_argument("--test-rollouts", type=int, default=500)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-variables", type=int, default=9)
    parser.add_argument("--invalid-reward", type=float, default=-100.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.val_rollouts < 1:
        raise ValueError("--val-rollouts must be >= 1")
    if args.test_rollouts < 1:
        raise ValueError("--test-rollouts must be >= 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    preferred = checkpoints_dir / args.checkpoint_name
    fallback_names = ["best_policy.pt", "best_model.pt"]
    checkpoint_path = preferred
    if not checkpoint_path.exists():
        for name in fallback_names:
            candidate = checkpoints_dir / name
            if candidate.exists():
                checkpoint_path = candidate
                break
    data_dir = Path(args.data_dir)
    out_path = run_dir / args.out_name

    check_paths(run_dir, checkpoint_path, data_dir, out_path)

    if args.check_only:
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_ids = available_equation_ids(data_dir)
    train_ids, val_ids, test_ids = split_equation_ids(
        all_ids,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=args.seed,
    )
    val_eqs = load_equation_datasets(val_ids, data_dir=data_dir)
    test_eqs = load_equation_datasets(test_ids, data_dir=data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dim, num_actions, hidden_dim = infer_policy_dims_from_checkpoint(checkpoint)

    policy = MLPPolicy(state_dim=state_dim, num_actions=num_actions, hidden_dim=hidden_dim).to(device)

    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    elif "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    else:
        policy.load_state_dict(checkpoint)

    policy.eval()

    run_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_file:
        evaluate_split(
            policy,
            val_eqs,
            "val",
            num_variables=args.num_variables,
            max_depth=args.max_depth,
            max_steps=args.max_steps,
            sample_size=args.sample_size,
            invalid_reward=args.invalid_reward,
            n_rollouts=args.val_rollouts,
            device=device,
            out_file=out_file,
        )
        evaluate_split(
            policy,
            test_eqs,
            "test",
            num_variables=args.num_variables,
            max_depth=args.max_depth,
            max_steps=args.max_steps,
            sample_size=args.sample_size,
            invalid_reward=args.invalid_reward,
            n_rollouts=args.test_rollouts,
            device=device,
            out_file=out_file,
        )

    print(f"Saved validation/test recovered expressions to: {out_path}")


if __name__ == "__main__":
    main()
