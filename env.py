from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data_loader import EquationDataset
from .grammar import Grammar, Node, apply_rule, get_sibling_symbols, tree_to_string
from .reward import evaluate_expression_reward


class SymbolicRegressionEnv(gym.Env):
    metadata = {"render_modes": []}

    # These operations are numerically unstable, so we exclude them
    unsafe_disabled = {"arcsin", "ln", "sqrt", "pow4", "pow5", "exp"}

    def __init__(
        self,
        grammar: Optional[Grammar] = None,
        equations: Optional[Sequence[EquationDataset]] = None,
        max_steps: int = 100,
        max_depth: int = 8,
        sample_size: int = 1000,
        invalid_reward: float = -100.0,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        formula_name: Optional[str] = None,
        num_variables: Optional[int] = None,
    ):
        super().__init__()

        # Allow passing raw X/y arrays directly instead of an EquationDataset
        if equations is None:
            if X is None or y is None:
                raise ValueError("Provide either equations or a single X/y dataset.")
            equations = [
                EquationDataset(
                    X=np.asarray(X, dtype=float),
                    y=np.asarray(y, dtype=float),
                    formula_name=formula_name or "manual_equation",
                )
            ]

        self.equations = list(equations)
        if not self.equations:
            raise ValueError("At least one equation dataset is required.")

        self.max_variables = num_variables or 9
        self.grammar = grammar or Grammar(num_variables=self.max_variables)
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.invalid_reward = invalid_reward

        self.num_actions = len(self.grammar.rules)

        # State = current symbol + depth + action mask + parent action + sibling symbols
        self.state_dim = (
            self.grammar.num_symbols
            + 1
            + self.num_actions
            + self.num_actions
            + self.grammar.num_symbols
        )

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.current_equation: Optional[EquationDataset] = None
        self.current_allowed_vars: set[str] = set()
        self.root: Optional[Node] = None
        self.holes: List[Node] = []
        self.past_actions: List[int] = []
        self.steps = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        super().reset(seed=seed)

        self.steps = 0
        self.root = Node("S", depth=0)
        self.holes = [self.root]
        self.past_actions = []

        # Pick a random equation for this episode
        if hasattr(self, "np_random"):
            index = int(self.np_random.integers(len(self.equations)))
        else:
            index = int(np.random.randint(len(self.equations)))
        self.set_equation(self.equations[index])

        return self._get_state(), {}

    def set_equation(self, equation: EquationDataset) -> None:
        self.current_equation = equation
        self.current_allowed_vars = set(equation.variable_names)

    def step(self, action: int):
        self.steps += 1
        valid_actions = self._get_valid_actions()

        # Penalise and stop immediately if an illegal action is chosen
        if action not in valid_actions:
            return (
                self._get_state(),
                self.invalid_reward,
                True,
                False,
                {
                    "expression": self._tree_to_expression(),
                    "invalid_action": True,
                    "equation": self._equation_name(),
                },
            )

        self._apply_action(action)
        self.past_actions.append(action)

        terminated = not self.holes
        truncated = self.steps >= self.max_steps

        # If we ran out of steps, close remaining holes with terminal symbols
        if truncated and self.holes:
            self._force_terminate()
            terminated = True

        done = terminated or truncated
        expression = self._tree_to_expression()
        reward_eval_seconds = 0.0

        if done:
            reward_t0 = time.perf_counter()
            reward = self._compute_reward(expression)
            reward_eval_seconds = time.perf_counter() - reward_t0
        else:
            reward = 0.0

        return (
            self._get_state(),
            reward,
            terminated,
            truncated,
            {
                "expression": expression,
                "expression_length": len(expression),
                "reward_eval_seconds": reward_eval_seconds,
                "equation": self._equation_name(),
                "past_actions": list(self.past_actions),
            },
        )

    def _force_terminate(self) -> None:
        # Fill each open hole with a valid terminal rule to complete the tree
        while self.holes:
            current_node = self.holes.pop(0)

            terminal_actions = self.grammar.get_terminal_only_actions(current_node.symbol)
            terminal_actions = [
                a for a in terminal_actions
                if not self._action_uses_invalid_variable(a)
            ]

            if not terminal_actions:
                continue

            action = int(np.random.choice(terminal_actions))
            rule = self.grammar.rules[action]
            apply_rule(current_node, rule.rhs, action_id=action)

            for child in reversed(current_node.children):
                if self.grammar.is_non_terminal(child.symbol):
                    self.holes.insert(0, child)

    def get_valid_actions(self) -> List[int]:
        return self._get_valid_actions()

    def _equation_name(self) -> Optional[str]:
        return None if self.current_equation is None else self.current_equation.formula_name

    def _get_state(self) -> np.ndarray:
        current_node = self.holes[0] if self.holes else None
        current_symbol = current_node.symbol if current_node is not None else None

        symbol_vec = self.grammar.symbol_to_onehot(current_symbol)

        # Normalise depth relative to the maximum allowed depth
        depth_value = 0.0 if current_node is None else current_node.depth / max(1, self.max_depth)
        depth_vec = np.array([depth_value], dtype=np.float32)

        action_mask = np.zeros(self.num_actions, dtype=np.float32)
        for action_id in self._get_valid_actions():
            action_mask[action_id] = 1.0

        parent_action_vec = np.zeros(self.num_actions, dtype=np.float32)
        if current_node is not None and current_node.parent_action is not None:
            parent_action_vec[current_node.parent_action] = 1.0

        sibling_vec = self.grammar.symbols_to_multihot(
            [] if current_node is None else get_sibling_symbols(current_node)
        )

        state = np.concatenate(
            [symbol_vec, depth_vec, action_mask, parent_action_vec, sibling_vec]
        )
        return state.astype(np.float32)

    def _get_valid_actions(self) -> List[int]:
        if not self.holes:
            return []

        current_node = self.holes[0]

        # At max depth, only terminal expansions are allowed
        if current_node.depth >= self.max_depth:
            valid = self.grammar.get_terminal_only_actions(current_node.symbol)
        else:
            valid = self.grammar.get_valid_actions(current_node.symbol)

        return [a for a in valid if not self._action_is_masked(a)]

    def _action_is_masked(self, action_id: int) -> bool:
        rule = self.grammar.rules[action_id]

        if rule.name in self.unsafe_disabled:
            return True

        if rule.rhs and rule.rhs[0] in self.unsafe_disabled:
            return True

        # Block variables that don't exist in the current equation
        if rule.name.startswith("var_"):
            var_name = rule.name.replace("var_", "")
            if var_name not in self.current_allowed_vars:
                return True

        return self._action_uses_invalid_variable(action_id)

    def _action_uses_invalid_variable(self, action_id: int) -> bool:
        if self.current_equation is None:
            return False

        num_allowed = self.current_equation.X.shape[1]
        rule = self.grammar.rules[action_id]

        for token in rule.rhs:
            if token.startswith("x") and token[1:].isdigit():
                if int(token[1:]) > num_allowed:
                    return True

        return False

    def _apply_action(self, action: int) -> None:
        current_node = self.holes.pop(0)
        rule = self.grammar.rules[action]

        if rule.lhs != current_node.symbol:
            raise ValueError(f"Action {action} does not expand symbol {current_node.symbol}.")

        apply_rule(current_node, rule.rhs, action_id=action)

        # Push new non-terminal children to the front of the hole queue
        for child in reversed(current_node.children):
            if self.grammar.is_non_terminal(child.symbol):
                self.holes.insert(0, child)

    def _is_done(self) -> bool:
        return not self.holes or self.steps >= self.max_steps

    def _tree_to_expression(self) -> str:
        if self.root is None:
            return ""
        return tree_to_string(self.root)

    def _compute_reward(self, expression: str) -> float:
        if self.current_equation is None:
            return self.invalid_reward

        inputs, targets = self.current_equation.sample_uniform(self.sample_size)

        # Equation I.6.2 only uses two variables; enforce this constraint
        required_vars = None
        equation_id = self.current_equation.equation_id or self.current_equation.formula_name
        if equation_id == "I.6.2":
            required_vars = ["x1", "x2"]

        return evaluate_expression_reward(
            expression,
            inputs,
            targets,
            invalid_reward=self.invalid_reward,
            required_vars=required_vars,
        )
