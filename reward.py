from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

import numpy as np
import sympy as sp


# Grammar non-terminals — their presence means the tree is incomplete
NON_TERMINAL_SYMBOLS = ("S", "E", "T", "F", "U", "A", "X", "C")

# Hard limits to reject expressions that are too large to evaluate safely
MAX_EXPRESSION_LENGTH = 200
MAX_EXPRESSION_TOKENS = 120
MAX_PARENTHESES_DEPTH = 24
MAX_SIMPLIFY_LENGTH   = 80
MAX_SIMPLIFY_TOKENS   = 60


# --- Helper math functions used inside eval() ---

def sq(values: np.ndarray) -> np.ndarray:
    return np.square(values)

def cube(values: np.ndarray) -> np.ndarray:
    return np.power(values, 3)

def pow4(values: np.ndarray) -> np.ndarray:
    return np.power(values, 4)

def pow5(values: np.ndarray) -> np.ndarray:
    return np.power(values, 5)

def neg(values: np.ndarray) -> np.ndarray:
    return -values


def _evaluation_context(inputs: np.ndarray) -> Dict[str, object]:
    # Build the namespace used by safe_eval, mapping variable names to data columns
    context: Dict[str, object] = {
        "np": np,
        "sq": sq, "cube": cube, "pow4": pow4, "pow5": pow5, "neg": neg,
        "sqrt": np.sqrt, "exp": np.exp,
        "sin": np.sin, "cos": np.cos, "tanh": np.tanh, "arcsin": np.arcsin,
        "ln": np.log,
        "pi": np.pi,
    }
    for col in range(inputs.shape[1]):
        context[f"x{col + 1}"] = inputs[:, col]
    return context


def safe_eval(expression: str, inputs: np.ndarray) -> np.ndarray:
    local_dict = _evaluation_context(np.asarray(inputs, dtype=float))

    # Suppress numpy warnings (overflow, divide-by-zero) during evaluation
    with np.errstate(all="ignore"):
        values = eval(expression, {"__builtins__": {}}, local_dict)

    values = np.asarray(values, dtype=float)

    # If the result is a scalar, broadcast it to match the number of data points
    if values.ndim == 0:
        values = np.full(inputs.shape[0], float(values), dtype=float)
    else:
        values = values.reshape(-1)

    return values


def _max_parentheses_depth(expression: str) -> int:
    depth, max_depth = 0, 0
    for token in expression:
        if token == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif token == ")":
            depth = max(0, depth - 1)
    return max_depth


def simplify_expr(expression: str) -> str:
    # Skip simplification for long expressions — SymPy can stall on them
    if len(expression) > MAX_SIMPLIFY_LENGTH:
        return expression

    rough_tokens = re.findall(r"[A-Za-z_]\w*|\d+\.\d+|\d+|[()+\-*/,]", expression)
    if len(rough_tokens) > MAX_SIMPLIFY_TOKENS:
        return expression

    # Rename sq/cube to pow2/pow3 so SymPy can parse them
    normalized = expression.replace("sq(", "pow2(").replace("cube(", "pow3(")

    symbols = {f"x{i}": sp.Symbol(f"x{i}") for i in range(1, 10)}
    local_dict = {
        **symbols,
        "pi": sp.pi,
        "pow2": lambda z: z**2,
        "pow3": lambda z: z**3,
        "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh,
        "neg": lambda z: -z,
    }

    try:
        sym_expr = sp.sympify(normalized, locals=local_dict)
        # Use sympify only, not simplify() — full simplification can hang training
        return str(sym_expr)
    except Exception:
        return expression


def evaluate_expression_reward(
    expression: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    invalid_reward: float = -100.0,
    required_vars: Optional[Sequence[str]] = None,
    max_expression_length: int = MAX_EXPRESSION_LENGTH,
    max_expression_tokens: int = MAX_EXPRESSION_TOKENS,
    max_parentheses_depth: int = MAX_PARENTHESES_DEPTH,
) -> float:
    # Reject incomplete expressions that still contain non-terminal symbols
    if any(symbol in expression for symbol in NON_TERMINAL_SYMBOLS):
        return invalid_reward

    if len(expression) > max_expression_length:
        return invalid_reward

    raw_tokens = re.findall(r"[A-Za-z_]\w*|\d+\.\d+|\d+|[()+\-*/,]", expression)
    if len(raw_tokens) > max_expression_tokens:
        return invalid_reward

    if _max_parentheses_depth(expression) > max_parentheses_depth:
        return invalid_reward

    simplified = simplify_expr(expression)

    if len(simplified) > max_expression_length:
        return invalid_reward

    try:
        predictions = safe_eval(simplified, inputs)
    except Exception:
        return invalid_reward

    targets = np.asarray(targets, dtype=float).reshape(-1)

    if predictions.shape[0] != targets.shape[0]:
        return invalid_reward
    if np.any(~np.isfinite(predictions)):
        return invalid_reward

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    if not np.isfinite(rmse):
        return invalid_reward

    # Reward = negative RMSE with a small complexity penalty for expression length
    expr_tokens = re.findall(r"[A-Za-z_]\w*|\d+\.\d+|\d+|[()+\-*/,]", simplified)
    complexity = len(expr_tokens)
    reward = -float(rmse + 0.001 * complexity)

    # Penalise trivial constant expressions
    if simplified in {"0", "1", "2", "pi"}:
        reward -= 0.5

    # Penalise missing required variables for equations that need them
    for var in required_vars or []:
        if var not in simplified:
            reward -= 0.2

    return float(np.clip(reward, invalid_reward, 0.0))
