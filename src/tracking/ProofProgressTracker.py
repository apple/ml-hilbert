#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from src.tools.lean_utils import extract_theorem_name
class ProofStatus(Enum):
    """Enumeration for different proof statuses."""
    PENDING = "PENDING"
    SOLVING = "SOLVING"
    SOLVED = "SOLVED"
    FAILED = "FAILED"

class ProofStrategy(Enum):
    """Enumeration for different proof strategies."""
    FORMAL_LLM = "FORMAL_LLM"
    SUBGOAL_DECOMP = "SUBGOAL_DECOMP"
    SHALLOW_SOLVE = "SHALLOW_SOLVE"
    RECURSIVE = "RECURSIVE"

@dataclass
class SubgoalNode:
    """Represents a subgoal in the proof tree."""
    name: str
    theorem: str
    status: ProofStatus = ProofStatus.PENDING
    strategy: Optional[ProofStrategy] = None
    depth: int = 0
    attempt: int = 0
    max_attempts: int = 1
    children: List['SubgoalNode'] = field(default_factory=list)
    proof: Optional[str] = None
    error_message: Optional[str] = None
    parent: Optional['SubgoalNode'] = None

@dataclass
class ProofTree:
    """Represents the entire proof tree structure."""
    root: SubgoalNode
    nodes_by_name: dict = field(default_factory=dict)
    max_depth: int = 3
    
    def __post_init__(self):
        """Initialize the node index after creation."""
        self.nodes_by_name[self.root.name] = self.root

    def print_tree(self) -> str:
        """Print the proof tree in a hierarchical format."""
        lines = []
        self._print_node(self.root, "", True, lines)
        return "\n".join(lines)
    
    def _print_node(self, node: SubgoalNode, prefix: str, is_last: bool, lines: List[str]) -> None:
        """Recursively print a node and its children."""
        # Determine the connector
        connector = "└─ " if is_last else "├─ "
        
        # Format the node information
        status_symbol = self._get_status_symbol(node.status)
        strategy_info = f" [{node.strategy.value}]" if node.strategy else ""
        depth_info = f" (depth {node.depth})" if node.depth > 0 else ""
        attempt_info = f" (attempt {node.attempt}/{node.max_attempts})" if node.max_attempts > 1 else ""
        
        # Truncate theorem for display
        theorem_display = node.theorem[:100] + "..." if len(node.theorem) > 60 else node.theorem
        theorem_display = theorem_display.replace('\n', ' ').strip()
        
        node_line = f"{prefix}{connector}{status_symbol} {node.name}: {theorem_display}{strategy_info}{depth_info}{attempt_info}"
        lines.append(node_line)
        
        # Add error message if present
        if node.error_message and node.status == ProofStatus.FAILED:
            error_prefix = prefix + ("    " if is_last else "│   ")
            error_line = f"{error_prefix}✗ Error: {node.error_message[:80]}..."
            lines.append(error_line)
        
        # Print children
        if node.children:
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self._print_node(child, child_prefix, is_last_child, lines)
    
    def _get_status_symbol(self, status: ProofStatus) -> str:
        """Get the symbol representing the proof status."""
        symbols = {
            ProofStatus.PENDING: "⏳",
            ProofStatus.SOLVING: "🔄",
            ProofStatus.SOLVED: "✅", 
            ProofStatus.FAILED: "❌"
        }
        return symbols.get(status, "?")
    
    def update_node_status(self, node_name: str, status: ProofStatus) -> None:
        """Update the status of the specified node."""
        if node_name in self.nodes_by_name:
            self.nodes_by_name[node_name].status = status
    
    def set_node_strategy(self, node_name: str, strategy: ProofStrategy) -> None:
        """Set the strategy for the specified node."""
        if node_name in self.nodes_by_name:
            self.nodes_by_name[node_name].strategy = strategy
    
    def add_subgoals(self, parent_name: str, subgoal_theorems: List[str]) -> List[SubgoalNode]:
        """Add subgoals to the specified parent node."""
        if parent_name not in self.nodes_by_name:
            return []
        
        parent_node = self.nodes_by_name[parent_name]
        new_nodes = []
        for i, theorem in enumerate(subgoal_theorems):
            # Extract theorem name for display
            theorem_name = extract_theorem_name(theorem)
            if not theorem_name:
                theorem_name = f"subgoal_{i+1}"
            
            subgoal = SubgoalNode(
                name=theorem_name,
                theorem=theorem,
                depth=parent_node.depth + 1,
                parent=parent_node
            )
            parent_node.children.append(subgoal)
            new_nodes.append(subgoal)
            # Add to name index
            self.nodes_by_name[theorem_name] = subgoal
        
    
    def get_node(self, name: str) -> Optional[SubgoalNode]:
        """Get a node by its name."""
        return self.nodes_by_name.get(name)
    
    def get_parent_node(self, node_name: str) -> Optional[SubgoalNode]:
        """Get the parent node of the specified node."""
        if node_name in self.nodes_by_name:
            return self.nodes_by_name[node_name].parent
        return None
    
    def remove_children(self, node_name: str) -> None:
        """Update node status and remove children if status is FAILED."""
        if node_name not in self.nodes_by_name:
            return
        
        node = self.nodes_by_name[node_name]
        
        self._remove_node_children(node)
    
    def _remove_node_children(self, node: SubgoalNode) -> None:
        """Recursively remove all children of a node from the tree."""
        for child in node.children:
            # Recursively remove grandchildren first
            self._remove_node_children(child)
            # Remove child from nodes_by_name index
            if child.name in self.nodes_by_name:
                del self.nodes_by_name[child.name]
        
        # Clear the children list
        node.children.clear()