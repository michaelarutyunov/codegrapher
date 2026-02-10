# Implementation Plan: Named Fuzzy Edges + 1-Level MRO

**Status:** DRAFT - Pending Approval
**Author:** Claude + User
**Date:** 2025-02-10
**Target Files:** `src/codegrapher/graph.py` (primary), `tests/test_graph.py` (new)

---

## Table of Contents

1. [Overview](#overview)
2. [Feature 1: Named Fuzzy Edges](#feature-1-named-fuzzy-edges)
3. [Feature 2: 1-Level MRO](#feature-2-1-level-mro)
4. [Pre-Resolved Decisions](#pre-resolved-decisions)
5. [Edge Case Specifications](#edge-case-specifications)
6. [Test Cases](#test-cases)
7. [Migration Path](#migration-path)
8. [Implementation Checklist](#implementation-checklist)

---

## Overview

### Goals

1. **Named Fuzzy Edges**: Replace `<unknown>.self.B1` with contextual names like `A2.B1`
2. **1-Level MRO**: Resolve `self.B1()` to parent class when possible: `A1.B1`

### Non-Goals

- Full import resolution (deferred to future)
- Multiple inheritance MRO (use first parent only)
- Type inference for non-`self` calls
- Runtime behavior analysis

### Success Criteria

- [ ] All existing tests pass
- [ ] New test cases pass (15+ tests)
- [ ] Index rebuild completes without errors on codegrapher repo
- [ ] Query `*.B1` finds callers of methods named B1
- [ ] Dead code detection becomes possible

---

## Feature 1: Named Fuzzy Edges

### Current Behavior

```python
# Current (line 209 in graph.py)
callee_id = f"<unknown>.{callee_name}"
# Examples:
#   self.B1()      → <unknown>.self.B1
#   foo()          → <unknown>.foo
#   obj.method()   → <unknown>.obj.method
```

### Target Behavior

```python
# New behavior
callee_id = _make_contextual_callee_id(callee_name, caller_id)

# Examples:
#   self.B1() in c2.A2.B2      → c2.A2.B1
#   foo() in c2.A2.B2          → foo
#   obj.method() in c2.A2.B2   → obj.method
#   cls.method() in c2.A2.B2   → c2.A2.method
```

### Implementation Specification

#### New Function: `_make_contextual_callee_id()`

**Location:** `src/codegrapher/graph.py` (after line 287, before `_resolve_base_class_name`)

**Signature:**
```python
def _make_contextual_callee_id(callee_name: str, caller_id: str) -> str:
    """Generate a contextual callee ID from a call expression.

    Args:
        callee_name: The extracted callee name (e.g., "self.B1", "foo", "obj.method")
        caller_id: Full symbol ID of caller (e.g., "c2.A2.B2", "c2.standalone_func")

    Returns:
        Contextual callee ID:
        - For "self.XXX": "{module}.{class}.{XXX}" (e.g., "c2.A2.B1")
        - For "cls.XXX": "{module}.{class}.{XXX}" (e.g., "c2.A2.factory")
        - For direct calls: "{XXX}" (e.g., "foo")
        - For obj.method: "obj.method" (unchanged)

    Examples:
        >>> _make_contextual_callee_id("self.B1", "c2.A2.B2")
        "c2.A2.B1"
        >>> _make_contextual_callee_id("foo", "c2.A2.B2")
        "foo"
        >>> _make_contextual_callee_id("obj.method", "c2.A2.B2")
        "obj.method"
    """
```

**Logic - EXACT Algorithm:**

```
1. If callee_name starts with "self.":
   a. Parse caller_id: ["module", "class", "method"] or ["module", "function"]
   b. If caller_id has 3+ parts (it's a method):
      - Extract module = parts[:-2].join(".")
      - Extract class = parts[-2]
      - Extract method = callee_name[5:]  # strip "self."
      - Return f"{module}.{class}.{method}"
   c. Else (function calling self mistakenly - invalid Python):
      - Return f"<invalid>.{callee_name}"

2. Elif callee_name starts with "cls.":
   a. Same logic as "self."
   b. Return f"{module}.{class}.{method_name}"

3. Else:
   a. Return callee_name as-is (direct calls, obj.method, etc.)
```

**Decision Point 1.1:** What if a function (not method) uses `self`?
- **DECISION:** Tag as `<invalid>.self.XXX` - this is syntactically invalid Python

**Decision Point 1.2:** What if `caller_id` is a module-level function?
- **DECISION:** For `self.` in non-method context → `<invalid>.self.XXX`

**Decision Point 1.3:** What about `super().method()`?
- **DECISION:** Keep as-is (`super().method`) - Feature 2 will handle this

#### Modified Function: `_extract_function_calls()`

**Location:** `src/codegrapher/graph.py`, line 178-220

**Change:**
```python
# OLD (line 206-212):
if callee_name:
    callee_id = f"<unknown>.{callee_name}"
    edges.append(Edge(caller_id=caller_id, callee_id=callee_id, type="call"))

# NEW:
if callee_name:
    callee_id = _make_contextual_callee_id(callee_name, caller_id)
    edges.append(Edge(caller_id=caller_id, callee_id=callee_id, type="call"))
```

**No other changes to this function.**

---

## Feature 2: 1-Level MRO

### Goal

Resolve `self.B1()` calls to the actual defining class using 1-level inheritance lookup.

### Current Behavior (after Feature 1)

```python
# After Feature 1:
self.B1() in c2.A2.B2 (where B1 is in A1) → c2.A2.B1
```

### Target Behavior

```python
# After Feature 2:
self.B1() in c2.A2.B2 (where B1 is in A1) → c1.A1.B1  # RESOLVED!
```

### Implementation Specification

#### New Helper Data Structure: `ClassHierarchyMap`

**Location:** New class in `src/codegrapher/graph.py` (before `extract_edges_from_file`)

**Purpose:** Cache class → methods and class → parent mappings for a file

```python
@dataclass
class ClassHierarchyMap:
    """In-memory mapping of class hierarchy for a single file.

    Built during edge extraction and used for 1-level MRO resolution.

    Attributes:
        class_methods: Dict mapping class_id → set of method names
                     Example: {"c2.A2": {"B2", "other"}, "c1.A1": {"B1"}}
        class_parents: Dict mapping class_id → list of parent class IDs
                     Example: {"c2.A2": ["c1.A1"], "c1.A1": []}
    """
    class_methods: Dict[str, Set[str]]
    class_parents: Dict[str, List[str]]

    def get_method_defining_class(self, class_id: str, method_name: str) -> Optional[str]:
        """Find which class defines a method using 1-level lookup.

        Search order:
        1. Check if method is in current class
        2. Check first parent (immediate base class only)
        3. Return None if not found

        Args:
            class_id: Full class ID (e.g., "c2.A2")
            method_name: Method name (e.g., "B1")

        Returns:
            Class ID that defines the method, or None if not found
        """
        # 1. Check current class
        if method_name in self.class_methods.get(class_id, set()):
            return class_id

        # 2. Check first parent only (1-level MRO)
        parents = self.class_parents.get(class_id, [])
        if parents and method_name in self.class_methods.get(parents[0], set()):
            return parents[0]

        # 3. Not found
        return None
```

**Decision Point 2.1:** Multiple inheritance - which parent to check?
- **DECISION:** Only check `parents[0]` (first parent). This covers ~80% of cases.

**Decision Point 2.2:** What if parent list is empty?
- **DECISION:** Return None - fall back to fuzzy resolution

**Decision Point 2.3:** What if parent is not in current file?
- **DECISION:** `class_parents` will have the ID from inherit edge (e.g., "c1.A1")
  - We can check if that class_id exists in `class_methods`
  - If not (cross-file), return None - fall back to fuzzy

#### New Function: `_build_class_hierarchy_map()`

**Location:** `src/codegrapher/graph.py` (after `ClassHierarchyMap` definition)

**Signature:**
```python
def _build_class_hierarchy_map(
    symbols: List[Symbol],
    inherit_edges: List[Edge]
) -> ClassHierarchyMap:
    """Build a class hierarchy map for 1-level MRO resolution.

    Args:
        symbols: All symbols extracted from the current file
        inherit_edges: All inheritance edges extracted from the current file

    Returns:
        ClassHierarchyMap with class → methods and class → parents mappings

    Notes:
        - Only includes classes defined in THIS file
        - Cross-file parent references are kept in class_parents but won't
          have entries in class_methods (handled gracefully by get_method_defining_class)
    """
```

**Logic - EXACT Algorithm:**

```
1. Initialize empty ClassHierarchyMap

2. For each symbol in symbols:
   a. If symbol.id has format "module.className.methodName":
      - class_id = ".".join(parts[:-1])  # "module.className"
      - method_name = parts[-1]
      - Add method_name to class_methods[class_id]

3. For each edge in inherit_edges:
   a. If edge.type == "inherit":
      - child_id = edge.caller_id  # "c2.A2"
      - parent_id = edge.callee_id  # "c1.A1"
      - Append parent_id to class_parents[child_id]

4. Return ClassHierarchyMap
```

**Decision Point 2.4:** What about standalone functions in the map?
- **DECISION:** Only add methods (symbols with 3+ ID parts) to class_methods

#### New Function: `_resolve_method_with_1level_mro()`

**Location:** `src/codegrapher/graph.py` (after `_build_class_hierarchy_map`)

**Signature:**
```python
def _resolve_method_with_1level_mro(
    callee_name: str,
    caller_id: str,
    hierarchy_map: ClassHierarchyMap,
    repo_root: Path,
    file_path: Path
) -> str:
    """Resolve a method call using 1-level inheritance lookup.

    Args:
        callee_name: Method call like "B1" (without self.) or "self.B1"
        caller_id: Full caller symbol ID (e.g., "c2.A2.B2")
        hierarchy_map: ClassHierarchyMap for the current file
        repo_root: Repository root path
        file_path: Current file being analyzed

    Returns:
        Resolved callee ID (e.g., "c1.A1.B1") or fallback fuzzy ID

    Examples:
        >>> _resolve_method_with_1level_mro("B1", "c2.A2.B2", hierarchy, ...)
        "c1.A1.B1"  # Found in parent class

        >>> _resolve_method_with_1level_mro("Zebra", "c2.A2.B2", hierarchy, ...)
        "c2.A2.Zebra"  # Not found, return contextual fuzzy
    """
```

**Logic - EXACT Algorithm:**

```
Input: callee_name may be "self.B1" or just "B1"
1. Extract method_name:
   a. If callee_name starts with "self.": method_name = callee_name[5:]
   b. Else: method_name = callee_name

2. Parse caller_id to get class_id:
   a. parts = caller_id.split(".")
   b. If len(parts) < 3: return f"<unknown>.{callee_name}"  # Not a method
   c. class_id = ".".join(parts[:-1])  # "c2.A2"

3. Try 1-level resolution:
   a. defining_class = hierarchy_map.get_method_defining_class(class_id, method_name)
   b. If defining_class is not None:
      return f"{defining_class}.{method_name}"

4. Fallback to contextual fuzzy:
   a. return f"{class_id}.{method_name}"
```

**Decision Point 2.5:** What if caller is a standalone function (not in a class)?
- **DECISION:** Return `<unknown>.{callee_name}` - can't do MRO on functions

**Decision Point 2.6:** What about `cls.method()` calls?
- **DECISION:** Same logic as `self.method()` - use the class from caller_id

#### Modified Function: `extract_edges_from_file()`

**Location:** `src/codegrapher/graph.py`, line 28-125

**Changes Required:**

```python
# ADD after line 62 (after symbol_ids dict):

# Build class hierarchy map for 1-level MRO (Feature 2)
hierarchy_map = _build_class_hierarchy_map(symbols, [])
# Note: inherit_edges not yet extracted, will update after

# MODIFY the AST walk to capture class context:

# Current (lines 64-119):
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        current_symbol_id = ...
    elif isinstance(node, ast.ClassDef):
        current_symbol_id = ...
        # Extract inheritance edges
        ...
    if current_symbol_id:
        call_edges = _extract_function_calls(node, current_symbol_id)

# NEW APPROACH - Two-pass:
# Pass 1: Extract class definitions and build hierarchy
# Pass 2: Extract edges with MRO resolution
```

**Revised `extract_edges_from_file()` Algorithm:**

```
1. Parse source → AST

2. Build symbol lookup (existing)

3. NEW: Build class hierarchy map
   - First pass: find all ClassDef nodes
   - Extract class info and build hierarchy_map
   - Extract inherit edges

4. NEW: Second pass: extract call edges with MRO
   - Walk AST
   - For each Call node within a symbol:
     - Try _resolve_method_with_1level_mro()
     - Create Edge with resolved callee_id

5. Extract import edges (existing)

6. Return all edges
```

**Decision Point 2.7:** Should we refactor the entire `extract_edges_from_file()`?
- **DECISION:** Yes, two-pass approach is cleaner. Refactor the function.

---

## Pre-Resolved Decisions

### Architectural Decisions

| ID | Question | Decision | Rationale |
|----|----------|----------|-----------|
| D1 | Modify Edge model? | **NO** | Keep callee_id as string, support new formats |
| D2 | Add confidence field to Edge? | **NO** | Not needed for v1.5, add later if needed |
| D3 | Database schema changes? | **NO** | Edge table stays the same |
| D4 | Backward compatibility? | **YES** | Keep fuzzy fallback for unresolvable cases |
| D5 | Breaking change for queries? | **NO** | Old fuzzy edges still work, new edges are additive |

### Coding Style Decisions

| ID | Question | Decision |
|----|----------|----------|
| S1 | Type hints for new functions? | **YES** - Use existing style (Optional, List, Dict) |
| S2 | Docstring format? | **YES** - Google style (existing in codebase) |
| S3 | Private function naming? | **YES** - Prefix with `_` |
| S4 | Dataclass for new types? | **YES** - Use @dataclass (matches Edge style) |
| S5 | Error handling style? | **YES** - Return fallback value, don't raise |

### Performance Decisions

| ID | Question | Decision |
|----|----------|----------|
| P1 | Cache hierarchy_map across calls? | **NO** - Built fresh per file (fast enough) |
| P2 | Lazy evaluation of MRO? | **NO** - Eager resolution during indexing |
| P3 | Memory overhead limit? | **YES** - hierarchy_map is per-file, discarded after use |

---

## Edge Case Specifications

### Edge Case 1: Nested Classes

```python
class Outer:
    class Inner:
        def foo(self):
            self.bar()  # Which class?
```

**Specification:**
- `caller_id` = `module.Outer.Inner.foo`
- Extracted class = `module.Outer.Inner`
- Resolution: Check `module.Outer.Inner` for `bar`, then parent
- **Expected:** `module.Outer.Inner.bar` (fuzzy if not found)

### Edge Case 2: Functions Using `self` (Invalid Python)

```python
def standalone():
    self.method()  # Would fail at runtime
```

**Specification:**
- `caller_id` = `module.standalone`
- Not a method (len(parts) < 3)
- **Expected:** `<invalid>.self.method`

### Edge Case 3: Multiple Inheritance

```python
class A:
    def method(self): pass

class B:
    def method(self): pass

class C(A, B):  # Two parents!
    def foo(self):
        self.method()  # Which one?
```

**Specification:**
- hierarchy_map.class_parents["module.C"] = ["module.A", "module.B"]
- 1-level lookup only checks `module.A` (first parent)
- **Expected:** `module.A.method` (Note: Not the full MRO, but "good enough")

### Edge Case 4: Cross-File Inheritance

```python
# c1.py
class A1:
    def B1(self): pass

# c2.py
from c1 import A1

class A2(A1):
    def B2(self):
        self.B1()  # Defined in c1.A1
```

**Specification:**
- Inherit edge: `c2.A2` → `c1.A1` (already works)
- hierarchy_map.class_methods["c2.A2"] = {"B2"}
- hierarchy_map.class_methods["c1.A1"] = NOT in map (different file!)
- 1-level lookup won't find `c1.A1` in class_methods
- **Expected:** Falls back to `c2.A2.B1` (contextual fuzzy)

**Note:** This is a known limitation. Full cross-file resolution is Feature 3.

### Edge Case 5: `super()` Calls

```python
class A2(A1):
    def B2(self):
        super().B1()  # Different from self.B1()
```

**Specification:**
- `callee_name` = `super().B1` (from `_unparse_attribute`)
- Doesn't start with "self." or "cls."
- **Expected:** `super().B1` (unchanged, treated as direct call)

**Future:** Could enhance to resolve `super()` in later version.

### Edge Case 6: Class Methods and `cls`

```python
class A2:
    @classmethod
    def factory(cls):
        cls.build()  # cls, not self
```

**Specification:**
- `callee_name` = `cls.build`
- Starts with "cls." → same logic as "self."
- **Expected:** `module.A2.build`

### Edge Case 7: Static Methods

```python
class A2:
    @staticmethod
    def helper():
        A2.other()  # Direct class reference
```

**Specification:**
- `callee_name` = `A2.other`
- Doesn't start with "self."
- **Expected:** `A2.other` (unchanged)

### Edge Case 8: Lambda and Nested Functions

```python
class A2:
    def method(self):
        def inner():
            self.foo()  # self from enclosing scope
```

**Specification:**
- AST walk will find the Call node
- `current_symbol_id` should be `module.A2.method` (not `inner`)
- **Expected:** `module.A2.foo` (resolved through enclosing method's class)

**Note:** Current implementation may not handle this correctly. File as TODO.

### Edge Case 9: Property Decorators

```python
class A2:
    @property
    def prop(self):
        return self.value  # self access in property
```

**Specification:**
- Treated same as regular method
- **Expected:** `module.A2.value`

### Edge Case 10: Dunder Methods

```python
class A2:
    def __init__(self):
        self.__str__()  # Calling own dunder
```

**Specification:**
- Dunder methods are regular methods
- **Expected:** `module.A2.__str__` (if defined in A2)

### Edge Case 11: Method Name Conflicts

```python
class A1:
    def process(self): pass

class A2(A1):
    def process(self): pass  # Override
    def call(self):
        self.process()  # Calls A2.process, not A1.process
```

**Specification:**
- hierarchy_map.class_methods["module.A2"] = {"process", "call"}
- 1-level lookup finds `process` in `module.A2` first
- **Expected:** `module.A2.process` (correct - uses override)

### Edge Case 12: Inheritance Chain > 2 Levels

```python
class A: pass
class B(A): pass
class C(B): pass
# C inherits from B, B inherits from A
```

**Specification:**
- Inherit edges: `C → B`, `B → A`
- For `self.method()` in C:
  - Check C for method (not found)
  - Check B (first parent, not found)
  - **Expected:** `module.B.method` (fuzzy, doesn't find A)
- **Note:** Only 1-level, not full chain

### Edge Case 13: No Base Class

```python
class A2:  # No (object) in Python 3
    def method(self):
        self.foo()
```

**Specification:**
- Inherit edge list is empty
- 1-level lookup returns None
- **Expected:** `module.A2.foo` (contextual fuzzy)

### Edge Case 14: Dynamic Base Class

```python
Base = SomeClass

class A2(Base):  # Variable as base class
    pass
```

**Specification:**
- AST base is `ast.Name` with id="Base"
- `_resolve_base_class_name` returns `module.Base`
- Inherit edge: `module.A2` → `module.Base`
- **Expected:** Edge is created, but `module.Base` won't be in class_methods
- Lookup falls back to fuzzy

### Edge Case 15: Empty File / Syntax Error

**Specification:**
- Existing behavior: Return empty edges list
- No change needed

---

## Test Cases

### Test File Structure

```python
# tests/test_graph.py
"""Unit tests for call graph construction and edge extraction."""

import pytest
from pathlib import Path
from codegrapher.graph import (
    extract_edges_from_file,
    _make_contextual_callee_id,
    _build_class_hierarchy_map,
    _resolve_method_with_1level_mro,
    ClassHierarchyMap,
)
from codegrapher.parser import extract_symbols
from codegrapher.models import Edge
```

### Test Suite 1: Named Fuzzy Edges (Feature 1)

```python
class TestMakeContextualCalleeId:
    """Test _make_contextual_callee_id function."""

    def test_self_method_in_class(self):
        """self.B1() in class method becomes module.class.B1"""
        result = _make_contextual_callee_id("self.B1", "c2.A2.B2")
        assert result == "c2.A2.B1"

    def test_self_method_top_level_function(self):
        """self.X in function is invalid"""
        result = _make_contextual_callee_id("self.X", "module.func")
        assert result == "<invalid>.self.X"

    def test_direct_function_call(self):
        """foo() stays as foo"""
        result = _make_contextual_callee_id("foo", "c2.A2.B2")
        assert result == "foo"

    def test_obj_method_call(self):
        """obj.method() stays as obj.method"""
        result = _make_contextual_callee_id("obj.method", "c2.A2.B2")
        assert result == "obj.method"

    def test_cls_method_in_class(self):
        """cls.build() becomes module.class.build"""
        result = _make_contextual_callee_id("cls.build", "c2.A2.factory")
        assert result == "c2.A2.build"

    def test_nested_class_self_call(self):
        """self.bar() in nested class uses full class path"""
        result = _make_contextual_callee_id("self.bar", "pkg.Outer.Inner.foo")
        assert result == "pkg.Outer.Inner.bar"

    def test_super_call_unchanged(self):
        """super().method() stays as-is"""
        result = _make_contextual_callee_id("super().method", "c2.A2.B2")
        assert result == "super().method"
```

### Test Suite 2: Class Hierarchy Map (Feature 2)

```python
class TestBuildClassHierarchyMap:
    """Test _build_class_hierarchy_map function."""

    @pytest.fixture
    def sample_symbols(self):
        """Create sample symbols for testing."""
        # Would use actual Symbol objects
        pass

    @pytest.fixture
    def sample_inherit_edges(self):
        """Create sample inheritance edges."""
        return [
            Edge("c2.A2", "c1.A1", "inherit"),
            Edge("c1.A1", "object", "inherit"),
        ]

    def test_extracts_class_methods(self, sample_symbols):
        """Map should contain class → methods mapping."""
        hierarchy = _build_class_hierarchy_map(sample_symbols, [])
        assert "c2.A2" in hierarchy.class_methods
        assert "B2" in hierarchy.class_methods["c2.A2"]

    def test_extracts_parent_relationships(self, sample_symbols, sample_inherit_edges):
        """Map should contain class → parents mapping."""
        hierarchy = _build_class_hierarchy_map(sample_symbols, sample_inherit_edges)
        assert hierarchy.class_parents["c2.A2"] == ["c1.A1"]

    def test_ignores_standalone_functions(self, sample_symbols):
        """Map should not include standalone functions."""
        hierarchy = _build_class_hierarchy_map(sample_symbols, [])
        assert "module.func" not in hierarchy.class_methods

    def test_handles_multiple_inheritance(self):
        """Map should store all parents."""
        edges = [
            Edge("c.C", "c.A", "inherit"),
            Edge("c.C", "c.B", "inherit"),
        ]
        hierarchy = _build_class_hierarchy_map([], edges)
        assert hierarchy.class_parents["c.C"] == ["c.A", "c.B"]

    def test_empty_inputs(self):
        """Empty symbols/edges should produce empty map."""
        hierarchy = _build_class_hierarchy_map([], [])
        assert len(hierarchy.class_methods) == 0
        assert len(hierarchy.class_parents) == 0
```

### Test Suite 3: 1-Level MRO Resolution (Feature 2)

```python
class TestResolveMethodWith1LevelMRO:
    """Test _resolve_method_with_1level_mro function."""

    @pytest.fixture
    def sample_hierarchy(self):
        """Create a sample class hierarchy."""
        return ClassHierarchyMap(
            class_methods={
                "c2.A2": {"B2", "other"},
                "c1.A1": {"B1", "A1_method"},
            },
            class_parents={
                "c2.A2": ["c1.A1"],
                "c1.A1": [],
            }
        )

    def test_resolves_method_in_current_class(self, sample_hierarchy):
        """Method defined in current class."""
        result = _resolve_method_with_1level_mro(
            "self.B2", "c2.A2.B2", sample_hierarchy, repo_root, file_path
        )
        assert result == "c2.A2.B2"

    def test_resolves_inherited_method(self, sample_hierarchy):
        """Method defined in parent class."""
        result = _resolve_method_with_1level_mro(
            "self.B1", "c2.A2.B2", sample_hierarchy, repo_root, file_path
        )
        assert result == "c1.A1.B1"

    def test_falls_back_to_fuzzy_for_unknown_method(self, sample_hierarchy):
        """Unknown method falls back to contextual fuzzy."""
        result = _resolve_method_with_1level_mro(
            "self.Zebra", "c2.A2.B2", sample_hierarchy, repo_root, file_path
        )
        assert result == "c2.A2.Zebra"

    def test_handles_standalone_function_caller(self, sample_hierarchy):
        """Standalone function can't use MRO."""
        result = _resolve_method_with_1level_mro(
            "foo", "module.func", sample_hierarchy, repo_root, file_path
        )
        assert result == "<unknown>.foo"

    def test_checks_first_parent_only(self):
        """Multiple inheritance: only check first parent."""
        hierarchy = ClassHierarchyMap(
            class_methods={
                "c.C": {},
                "c.A": {"method"},
                "c.B": {"method"},  # Also has method
            },
            class_parents={"c.C": ["c.A", "c.B"]}
        )
        result = _resolve_method_with_1level_mro(
            "self.method", "c.C.foo", hierarchy, repo_root, file_path
        )
        assert result == "c.A.method"  # First parent, not c.B.method

    def test_handles_cls_method_call(self, sample_hierarchy):
        """cls.method() should resolve like self.method()."""
        result = _resolve_method_with_1level_mro(
            "cls.B1", "c2.A2.factory", sample_hierarchy, repo_root, file_path
        )
        assert result == "c1.A1.B1"
```

### Test Suite 4: Integration Tests

```python
class TestExtractEdgesFromFile:
    """Integration tests for extract_edges_from_file."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with test files."""
        # Create test Python files
        (tmp_path / "c1.py").write_text("""
class A1:
    def B1(self):
        return "base"
""")
        (tmp_path / "c2.py").write_text("""
from c1 import A1

class A2(A1):
    def B2(self):
        result = self.B1()  # Should resolve to c1.A1.B1
        return result

    def direct_call(self):
        return foo()  # Should be "foo"
""")
        return tmp_path

    def test_self_call_resolves_to_parent_class(self, temp_repo):
        """self.B1() in A2 should resolve to c1.A1.B1."""
        c2_path = temp_repo / "c2.py"
        symbols = extract_symbols(c2_path, temp_repo)
        edges = extract_edges_from_file(c2_path, temp_repo, symbols)

        # Find the call edge from B2 to B1
        call_edges = [e for e in edges if e.type == "call"]
        b2_calls = [e for e in call_edges if e.caller_id == "c2.A2.B2"]

        # Should have edge to c1.A1.B1 (or contextual fuzzy if cross-file)
        assert any(e.callee_id in ["c1.A1.B1", "c2.A2.B1"] for e in b2_calls)

    def test_direct_call_unchanged(self, temp_repo):
        """Direct foo() call should stay as "foo"."""
        c2_path = temp_repo / "c2.py"
        symbols = extract_symbols(c2_path, temp_repo)
        edges = extract_edges_from_file(c2_path, temp_repo, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        assert any(e.callee_id == "foo" for e in call_edges)

    def test_inherit_edge_extracted(self, temp_repo):
        """Inheritance edge should be extracted."""
        c2_path = temp_repo / "c2.py"
        symbols = extract_symbols(c2_path, temp_repo)
        edges = extract_edges_from_file(c2_path, temp_repo, symbols)

        inherit_edges = [e for e in edges if e.type == "inherit"]
        assert any(e.caller_id == "c2.A2" and e.callee_id == "c1.A1"
                   for e in inherit_edges)

    def test_no_edges_for_syntax_error(self, tmp_path):
        """Syntax error should return empty edges."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("class : (invalid syntax")

        symbols = []  # No symbols from bad file
        edges = extract_edges_from_file(bad_file, tmp_path, symbols)
        assert edges == []
```

### Test Suite 5: Edge Case Tests

```python
class TestEdgeCases:
    """Test specific edge cases."""

    def test_nested_classes(self, tmp_path):
        """Nested class self calls."""
        # (implementation per edge case #1)

    def test_invalid_self_in_function(self, tmp_path):
        """self in standalone function."""
        # (implementation per edge case #2)

    def test_multiple_inheritance_first_parent(self, tmp_path):
        """Multiple inheritance uses first parent."""
        # (implementation per edge case #3)

    def test_super_call_unchanged(self, tmp_path):
        """super().method() stays unchanged."""
        # (implementation per edge case #5)

    def test_classmethod_cls_call(self, tmp_path):
        """cls.method() in classmethod."""
        # (implementation per edge case #6)
```

---

## Migration Path

### Phase 1: Implementation (No Breaking Changes)

1. **Add new functions** to `graph.py`
   - `_make_contextual_callee_id()`
   - `ClassHierarchyMap` class
   - `_build_class_hierarchy_map()`
   - `_resolve_method_with_1level_mro()`

2. **Modify existing functions**
   - `_extract_function_calls()` - use new callee ID generation
   - `extract_edges_from_file()` - refactor to two-pass

3. **Add tests** in `tests/test_graph.py`
   - All test suites from above

4. **Verify** all existing tests still pass

### Phase 2: Rebuild Index

```bash
# Users need to rebuild their index after deploying this change
cd /path/to/repo
codegraph build --full
```

### Phase 3: Validation

```bash
# Test on codegrapher itself
cd /home/mikhailarutyunov/projects/codegrapher
codegraph build --full

# Query to verify new edge format
# (via MCP or direct DB inspection)
```

### Rollback Plan

If issues arise:

1. Revert `graph.py` changes
2. Rebuild index with old code
3. No database migration needed (edges table unchanged)

---

## Implementation Checklist

### Pre-Implementation

- [ ] Review this plan with user
- [ ] Approve all pre-resolved decisions
- [ ] Confirm edge case handling
- [ ] Set up branch for changes

### Feature 1: Named Fuzzy Edges

- [ ] Implement `_make_contextual_callee_id()`
  - [ ] Handle "self." prefix
  - [ ] Handle "cls." prefix
  - [ ] Handle invalid cases
  - [ ] Add docstring with examples
- [ ] Modify `_extract_function_calls()`
  - [ ] Replace fuzzy ID generation
  - [ ] Update docstring if needed
- [ ] Add tests for `_make_contextual_callee_id()`
  - [ ] Test suite 1: 8 test cases

### Feature 2: 1-Level MRO

- [ ] Implement `ClassHierarchyMap` dataclass
  - [ ] class_methods dict
  - [ ] class_parents dict
  - [ ] get_method_defining_class() method
- [ ] Implement `_build_class_hierarchy_map()`
  - [ ] Extract methods from symbols
  - [ ] Extract parents from edges
  - [ ] Handle cross-file references
- [ ] Implement `_resolve_method_with_1level_mro()`
  - [ ] Parse caller_id
  - [ ] Try 1-level lookup
  - [ ] Fallback to fuzzy
- [ ] Refactor `extract_edges_from_file()`
  - [ ] Two-pass approach
  - [ ] Build hierarchy in pass 1
  - [ ] Use MRO in pass 2
- [ ] Add tests
  - [ ] Test suite 2: hierarchy map (6 tests)
  - [ ] Test suite 3: MRO resolution (7 tests)
  - [ ] Test suite 4: integration (4 tests)
  - [ ] Test suite 5: edge cases (6+ tests)

### Validation & Documentation

- [ ] Run all existing tests → PASS
- [ ] Run new tests → PASS
- [ ] Rebuild codegrapher index
- [ ] Verify edge quality in DB
- [ ] Update relevant docstrings
- [ ] Add migration note to CHANGELOG

### Post-Implementation

- [ ] Test on sample codebase
- [ ] Measure performance impact
- [ ] Document known limitations (cross-file inheritance)
- [ ] Create examples of new query capabilities

---

## Known Limitations (Post-Implementation)

1. **Cross-file inheritance**: Not resolved (falls back to fuzzy)
2. **Multiple inheritance**: Only checks first parent
3. **Nested scopes**: May not handle `self` in nested functions correctly
4. **Dynamic base classes**: Variable base classes not resolved
5. **`super()` calls**: Not specially handled

**Future Work:** Feature 3 could address cross-file resolution using the import resolver.

---

## Appendix: File Change Summary

### Modified Files

| File | Lines Added | Lines Modified | Lines Deleted | Net Change |
|------|-------------|----------------|---------------|------------|
| `src/codegrapher/graph.py` | ~180 | ~20 | ~10 | ~+190 |

### New Files

| File | Purpose |
|------|---------|
| `tests/test_graph.py` | Unit tests for graph module (~350 lines) |

### Database Schema Changes

| Table | Changes |
|-------|---------|
| `edges` | NONE (callee_id format changes, schema unchanged) |

---

## Approval

- [ ] User has reviewed this plan
- [ ] All ambiguities resolved
- [ ] Ready to proceed with implementation

**Sign-off:**

```
User: ________________  Date: ________
```
