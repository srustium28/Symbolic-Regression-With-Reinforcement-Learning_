from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# Symbols that can be expanded further (non-terminals in the grammar)
NON_TERMINALS = {"S", "E", "T", "F", "U", "A", "C", "X"}


def create_grammar(num_variables: int = 3) -> Dict[str, List[List[str]]]:
    # Full grammar with all allowed productions
    variable_names = [f"x{i + 1}" for i in range(num_variables)]
    return {
        "S": [["E"]],
        "E": [["T"], ["T", "+", "E"], ["T", "-", "E"]],
        "T": [["F"], ["F", "*", "T"], ["F", "/", "T"]],
        "F": [
            ["sq", "(", "U", ")"],
            ["cube", "(", "U", ")"],
            ["U"],
        ],
        "U": [
            ["neg", "(", "A", ")"],
            ["sin", "(", "A", ")"],
            ["cos", "(", "A", ")"],
            ["tanh", "(", "A", ")"],
            ["A"],
        ],
        "A": [["X"], ["C"], ["(", "E", ")"]],
        "X": [[name] for name in variable_names],
        "C": [["1"], ["2"], ["pi"]],
    }


def create_terminating_rules(num_variables: int = 3) -> Dict[str, List[List[str]]]:
    # Simplified rules that always lead to a terminal — used when depth limit is reached
    variable_names = [f"x{i + 1}" for i in range(num_variables)]
    return {
        "S": [["E"]],
        "E": [["T"]],
        "T": [["F"]],
        "F": [["U"]],
        "U": [["A"]],
        "A": [["X"], ["C"]],
        "X": [[name] for name in variable_names],
        "C": [["1"], ["2"], ["pi"]],
    }


@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: tuple[str, ...]
    name: str = ""


@dataclass
class Node:
    symbol: str
    depth: int = 0
    children: List["Node"] = field(default_factory=list)
    parent: Optional["Node"] = None
    parent_action: Optional[int] = None

    def is_nonterminal(self) -> bool:
        return self.symbol in NON_TERMINALS

    def is_hole(self) -> bool:
        # A hole is an unexpanded non-terminal node
        return self.is_nonterminal() and not self.children


def find_leftmost_hole(node: Node) -> Optional[Node]:
    if node.is_hole():
        return node
    for child in node.children:
        hole = find_leftmost_hole(child)
        if hole is not None:
            return hole
    return None


def apply_rule(node: Node, rhs: Sequence[str], action_id: Optional[int] = None) -> None:
    # Expand a node by attaching its right-hand side tokens as children
    node.children = [
        Node(symbol=token, depth=node.depth + 1, parent=node, parent_action=action_id)
        for token in rhs
    ]


def get_sibling_symbols(node: Node) -> List[str]:
    if node.parent is None:
        return []
    return [child.symbol for child in node.parent.children if child is not node]


def get_valid_rules(
    symbol: str,
    depth: int,
    max_depth: int,
    grammar: Optional[Dict[str, List[List[str]]]] = None,
    terminating_rules: Optional[Dict[str, List[List[str]]]] = None,
) -> List[List[str]]:
    grammar = grammar or create_grammar()
    terminating_rules = terminating_rules or create_terminating_rules()
    # Switch to terminating rules once we hit the depth limit
    if depth >= max_depth:
        return terminating_rules[symbol]
    return grammar[symbol]


def tree_to_string(node: Node) -> str:
    if not node.children:
        return node.symbol
    return "".join(tree_to_string(child) for child in node.children)


def count_holes(node: Node) -> int:
    total = 1 if node.is_hole() else 0
    for child in node.children:
        total += count_holes(child)
    return total


class Grammar:
    def __init__(self, num_variables: int = 3):
        self.num_variables = num_variables
        self.productions = create_grammar(num_variables)
        self.terminating_productions = create_terminating_rules(num_variables)

        self.rules: List[Rule] = []
        self._actions_by_lhs: Dict[str, List[int]] = defaultdict(list)
        self._terminal_only_actions: Dict[str, List[int]] = defaultdict(list)

        # Pre-compute which productions always lead to a terminal
        terminal_rhs = {
            lhs: {tuple(rhs) for rhs in rhs_list}
            for lhs, rhs_list in self.terminating_productions.items()
        }

        # Register every production as a numbered rule (action)
        for lhs, rhs_list in self.productions.items():
            for rhs in rhs_list:
                action_id = len(self.rules)
                rhs_tuple = tuple(rhs)
                rule = Rule(lhs=lhs, rhs=rhs_tuple, name=self._rule_name(lhs, rhs_tuple))
                self.rules.append(rule)
                self._actions_by_lhs[lhs].append(action_id)
                if rhs_tuple in terminal_rhs[lhs]:
                    self._terminal_only_actions[lhs].append(action_id)

        # Collect all symbols (terminals and non-terminals) for one-hot encoding
        all_symbols = set(self.productions.keys())
        for rhs_list in self.productions.values():
            for rhs in rhs_list:
                all_symbols.update(rhs)

        self.symbols = sorted(all_symbols)
        self.symbol_to_index = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self.num_symbols = len(self.symbols)

    def is_non_terminal(self, symbol: str) -> bool:
        return symbol in NON_TERMINALS

    def get_valid_actions(self, symbol: str) -> List[int]:
        return list(self._actions_by_lhs.get(symbol, []))

    def get_terminal_only_actions(self, symbol: str) -> List[int]:
        return list(self._terminal_only_actions.get(symbol, []))

    def symbol_to_onehot(self, symbol: Optional[str]) -> np.ndarray:
        vector = np.zeros(self.num_symbols, dtype=np.float32)
        if symbol in self.symbol_to_index:
            vector[self.symbol_to_index[symbol]] = 1.0
        return vector

    def symbols_to_multihot(self, symbols: Sequence[str]) -> np.ndarray:
        vector = np.zeros(self.num_symbols, dtype=np.float32)
        for symbol in symbols:
            idx = self.symbol_to_index.get(symbol)
            if idx is not None:
                vector[idx] = 1.0
        return vector

    @staticmethod
    def _rule_name(lhs: str, rhs: tuple[str, ...]) -> str:
        # Name variable rules as var_x1, var_x2, etc. for easy masking later
        if lhs == "X" and len(rhs) == 1 and rhs[0].startswith("x"):
            return f"var_{rhs[0]}"
        if rhs and rhs[0] in {"arcsin", "ln", "sqrt", "pow4", "pow5", "exp"}:
            return rhs[0]
        return f"{lhs}__{'_'.join(rhs)}"
