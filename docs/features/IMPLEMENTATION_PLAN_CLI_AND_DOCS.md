# Implementation Plan: CLI + Documentation Updates

**Date:** 2025-02-10
**Status:** DRAFT - Pending Approval
**Related Issues:** Named Fuzzy Edges + 1-Level MRO Implementation

---

## Table of Contents

1. [Task 1: Update MCP Tool Description](#task-1-update-mcp-tool-description)
2. [Task 2: Add `callers` CLI Command](#task-2-add-callers-cli-command)
3. [Task 3: Update README](#task-3-update-readme)
4. [Pre-Resolved Decisions](#pre-resolved-decisions)
5. [Implementation Checklist](#implementation-checklist)

---

## Task 1: Update MCP Tool Description

### Goal

Update the `codegraph_query` MCP tool description to accurately reflect the new capabilities after named fuzzy edges + 1-level MRO implementation.

### Changes Required

#### 1.1 Add "Finding Method Callers" Section

**Location:** After line 663, under "When to use CodeGrapher"

```markdown
**Finding Method Callers:**
- "methods that call compute_pagerank"
- "what uses extract_edges_from_file"
- "classes calling Database.get_all_symbols"
```

#### 1.2 Update "Results Ranking" Section

**Location:** Lines 690-701

**Add after PageRank interpretation:**

```markdown
**Call Graph Resolution:**
- Same-file inheritance: Fully resolved (e.g., `Derived.method` → `Base.method`)
- Cross-file inheritance: Contextual fuzzy (e.g., `Derived.method` → `Derived.parent_method`)
- Direct calls: Clean format (e.g., "foo", "obj.method")
- Legacy fuzzy: `<unknown>.name` for unresolved edge cases (<5% of edges)
```

#### 1.3 Add New Examples

**Location:** After line 727

```markdown
# NEW: Finding method callers (improved resolution)
>>> codegraph_query(query="methods that call compute_pagerank")
# Returns: extract_edges_from_file, cli commands using it...

# NEW: Finding inheritance relationships
>>> codegraph_query(query="subclasses of Exception")
# Returns: SecretFoundError, classes that inherit from Exception

# NEW: Finding database interaction patterns
>>> codegraph_query(query="classes using method get_all_symbols")
# Returns: Server, CLI, any code that queries the database
```

#### 1.4 Update "Understanding Relationships" Example

**Current (line 663):**
> Understanding relationships: "what calls this function", "where is X used"

**Updated:**
> Understanding relationships: "what calls this function" (now with better resolution)
> Finding inheritance: "subclasses of Exception", "who inherits from BaseModel"
> Tracing usage: "where is TokenValidator used", "what uses Database class"

---

## Task 2: Add `callers` CLI Command

### Goal

Implement `codegraph callers <symbol>` command to find what calls a given symbol.

### Specification

#### 2.1 Command Signature

```bash
codegraph callers [-h] [--json] [--limit N] <symbol_id>
```

**Arguments:**
- `symbol_id`: The symbol to find callers for (e.g., `extract_edges_from_file`, `Database.get_all_symbols`)
- `--limit N`: Maximum number of callers to return (default: 20)
- `--json`: Output in JSON format
- `-h, --help`: Show help message

#### 2.2 Output Format

**Default (human-readable):**
```
Callers of extract_edges_from_file (found 3 callers):

1. cli.py:cmd_build (line 156)
   └─ extract_edges_from_file(repo_root, symbols)

2. indexer.py:IncrementalIndexer.update_file (line 89)
   └─ edges = extract_edges_from_file(path, symbols)

3. cli.py:cmd_query (line 314)
   └─ result = codegraph_query.fn(...)
```

**JSON format:**
```json
{
  "symbol_id": "extract_edges_from_file",
  "callers": [
    {
      "caller_id": "cli.py:cmd_build",
      "file": "cli.py",
      "line": 156,
      "caller_type": "function",
      "call_context": "extract_edges_from_file(repo_root, symbols)"
    }
  ],
  "total_callers": 3,
  "truncated": false
}
```

#### 2.3 Implementation Details

**File to modify:** `src/codegrapher/cli.py`

**New function:** `cmd_callers(args: argparse.Namespace)`

**Algorithm:**
```python
def cmd_callers(args: argparse.Namespace) -> None:
    """Find symbols that call the given symbol.

    Queries the edges table for call edges where callee_id matches
    the given pattern, then enriches results with symbol information.
    """
    # 1. Get database connection
    # 2. Query edges: SELECT * FROM edges WHERE type='call' AND callee_id LIKE ?
    # 3. Enrich with symbol information from symbols table
    # 4. Sort by relevance (PageRank score)
    # 5. Format and display
```

**SQL Query:**
```sql
SELECT
    e.caller_id,
    s.file,
    s.start_line,
    s.signature,
    pr.pagerank_score
FROM edges e
JOIN symbols s ON e.caller_id = s.id
LEFT JOIN pagerank pr ON s.id = pr.symbol_id
WHERE e.type = 'call'
  AND e.callee_id LIKE ?
ORDER BY pr.pagerank_score DESC
LIMIT ?
```

#### 2.4 Edge Resolution Handling

| Edge Type | Display Format | Example |
|-----------|----------------|---------|
| Resolved (same-file) | Show full path | `src.Class.method` |
| Contextual fuzzy | Show as-is | `src.Class.parent_method` |
| Legacy fuzzy | Show with warning | `<unknown>.method (unresolved)` |
| Direct call | Show clean | `function_name` |

#### 2.5 Error Handling

```python
# If no callers found:
print(f"No callers found for '{args.symbol}'")
print(f"Tip: Use the full symbol ID (e.g., 'module.Class.method')")

# If symbol not found:
print(f"Symbol '{args.symbol}' not found in index")
print(f"Did you mean one of these?")
# Suggest similar symbols based on edge callee_ids

# If index doesn't exist:
print("Error: CodeGrapher index not found")
print("Run 'codegraph build' to create the index")
```

#### 2.6 Integration with argparse

**Add to `create_parser()` function in cli.py:**

```python
subparsers = parser.add_subparsers(dest='command', required=True)

# Add callers subcommand
parser_callers = subparsers.add_parser(
    'callers',
    help='Find symbols that call the given symbol'
)
parser_callers.add_argument(
    'symbol',
    help='Symbol to find callers for (e.g., extract_edges_from_file)'
)
parser_callers.add_argument(
    '--limit',
    type=int,
    default=20,
    help='Maximum number of callers to return (default: 20)'
)
parser_callers.add_argument(
    '--json',
    action='store_true',
    help='Output in JSON format'
)
```

---

## Task 3: Update README

### Goal

Update README.md with:
1. Detailed description of new features
2. Comparison vs full resolution tools
3. Current implementation limitations
4. Future enhancements roadmap

### Structure

#### 3.1 New Section: "Call Graph Features"

**Location:** After "How It Works" section

```markdown
## Call Graph Features

### Named Fuzzy Edges (v1.5)

CodeGrapher now extracts call graph edges with contextual names instead of
generic `<unknown>` prefixes:

| Before | After |
|--------|-------|
| `<unknown>.self.B1` | `c2.A2.B1` (contextual) |
| `<unknown>.self.B1` | `c1.A1.B1` (resolved, same-file inheritance) |

This enables queries like:
- "methods that call B1" → finds actual callers
- "what uses Database.get_all_symbols" → shows usage patterns

### 1-Level Inheritance Resolution

For classes in the same file, CodeGrapher resolves inherited method calls:

```python
# c1.py
class Base:
    def helper(self): pass

class Derived(Base):
    def method(self):
        self.helper()  # → Resolved to Base.helper!
```

**Limitation:** Cross-file inheritance falls back to contextual fuzzy
(e.g., `c2.A2.B1` instead of `c1.A1.B1`). Full cross-file resolution
requires import analysis and is planned for v2.
```

#### 3.2 New Section: "Comparison vs Full Resolution Tools"

```markdown
## Comparison with Full Resolution Tools

| Feature | CodeGrapher v1.5 | Pyright/mypy | pyan |
|---------|-------------------|--------------|------|
| **Type inference** | None | Full | None |
| **Inheritance resolution** | 1-level, same-file | Full | Import-based |
| **Import resolution** | File-level only | Full | Full |
| **Call graph extraction** | AST + fuzzy | Semantic | AST |
| **Index build time** | ~3 min (1K files) | ~30 sec | ~10 sec |
| **Query speed** | ~50-100ms | Instant | Instant |
| **Use case** | Semantic search | Type checking | Visualization |

**When to use CodeGrapher:**
- Exploring unfamiliar codebases
- Semantic code search ("authentication logic")
- Understanding call patterns
- Quick navigation without IDE

**When to use full resolution tools:**
- Type checking before commits
- Finding type errors
- Complete accuracy required
- IDE integration (real-time analysis)
```

#### 3.3 New Section: "Current Limitations"

```markdown
## Current Limitations

### Known Constraints

1. **Cross-file inheritance**
   - Falls back to contextual fuzzy (e.g., `Derived.parent_method`)
   - Full resolution planned for v2

2. **Multiple inheritance**
   - Only checks first parent for 1-level MRO
   - Covers ~80% of single-inheritance cases

3. **Nested classes**
   - Parser extracts top-level definitions only
   - Inner classes not indexed

4. **Dynamic features**
   - No runtime behavior analysis
   - Decorators like `@property` not specially handled
   - Metaclasses not understood

5. **`super()` calls**
   - Partially handled (extracted as `super.method` not fully resolved)

### Edge Case Behavior

| Pattern | Resolution |
|---------|------------|
| `self.method()` in same class | `Class.method` (resolved) |
| `self.method()` (inherited) | `BaseClass.method` (resolved, same-file) |
| `self.method()` (cross-file) | `CurrentClass.method` (fuzzy) |
| `super().method()` | `super().method` (as-is) |
| `obj.method()` | `obj.method` (unchanged) |
| `cls.method()` | `Class.method` (resolved) |
```

#### 3.4 New Section: "Roadmap"

```markdown
## Roadmap

### Planned Enhancements

#### Near-term (v1.6-v1.7)

- [ ] **`codegraph callers` command** - Find what calls a symbol
- [ ] **`codegraph hierarchy` command** - Show class inheritance tree
- [ ] **`codegraph broken` command** - Find potential broken references
- [ ] **Import-aware edge resolution** - Resolve cross-file inheritance

#### Mid-term (v2.0)

- [ ] **Full cross-file resolution** - Resolve calls across files
- [ ] **Multiple inheritance MRO** - Follow Python's C3 linearization
- [ ] **`super()` call resolution** - Properly handle `super().method()`
- [ ] **Edge validation** - Detect calls to non-existent methods

#### Future Considerations

- [ ] **Type-aware resolution** - Use type hints for better accuracy
- [ ] **Decorator handling** - Properly handle `@property`, `@staticmethod`
- [ ] **Nested class support** - Index inner classes and methods
- [ ] **Incremental PageRank** - Update scores without full recomputation
```

---

## Pre-Resolved Decisions

### Documentation Decisions

| ID | Question | Decision | Rationale |
|----|----------|----------|-----------|
| D1 | Include comparison table? | **YES** - Helps users understand trade-offs | Transparency about capabilities |
| D2 | Mention Pyright/mypy? | **YES** - Acknowledge when to use other tools | Honest positioning |
| D3 | List all limitations? | **YES** - Set proper expectations | Avoid over-promising |
| D4 | Include roadmap? | **YES** - Shows direction | Users can see what's planned |

### CLI Command Decisions

| ID | Question | Decision | Rationale |
|----|----------|----------|-----------|
| C1 | Command name? | **`callers`** (not `find-callers`) | Consistent with future commands |
| C2 | JSON output format? | **YES** - Enables scripting | Automation support |
| C3 | Sort order? | **By PageRank score** | Most important callers first |
| C4 | Limit default? | **20** | Balances detail vs noise |
| C5 | Fuzzy matching symbol names? | **NO** - Require exact IDs | Avoid false positives |
| C6 | Show call context? | **YES** - First line of call | Helps understand usage |

### Error Handling Decisions

| ID | Question | Decision | Rationale |
|----|----------|----------|-----------|
| E1 | No callers found? | **Message + tip** | Helpful, not silent |
| E2 | Symbol not found? | **Suggest alternatives** | Guide user |
| E3 | Index doesn't exist? | **Clear error + fix** | Actionable feedback |
| E4 | Multiple matches? | **Show all** | Let user decide |
| E5 | Edge type display? | **Badge/label** | Visual clarity |

---

## Implementation Checklist

### Task 1: Update MCP Tool Description

- [ ] Add "Finding Method Callers" section to docstring
- [ ] Update "Results Ranking" with call graph resolution info
- [ ] Add new query examples (callers, inheritance, patterns)
- [ ] Update "Understanding relationships" example
- [ ] Verify description renders correctly in MCP client

### Task 2: Add `callers` CLI Command

- [ ] Create `cmd_callers()` function in cli.py
- [ ] Add `callers` subparser to argparse
- [ ] Implement SQL query for finding callers
- [ ] Enrich results with symbol information
- [ ] Implement human-readable output format
- [ ] Implement JSON output format
- [ ] Add error handling for no results
- [ ] Add error handling for symbol not found
- [ ] Add error handling for no index
- [ ] Sort results by PageRank score
- [ ] Add `--limit` parameter support
- [ ] Add `--json` parameter support
- [ ] Write unit tests for cmd_callers
- [ ] Update CLI help documentation
- [ ] Test on codegrapher codebase

### Task 3: Update README

- [ ] Add "Call Graph Features" section
- [ ] Add comparison table vs full resolution tools
- [ ] Add "Current Limitations" section
- [ ] Add "Roadmap" section
- [ ] Update feature list in header
- [ ] Ensure all examples are accurate
- [ ] Verify links and formatting

---

## Appendix: File Changes Summary

| File | Lines Added | Lines Modified | Purpose |
|------|-------------|----------------|---------|
| `src/codegrapher/server.py` | ~50 | ~10 | Update MCP tool docstring |
| `src/codegrapher/cli.py` | ~150 | ~10 | Add callers command |
| `README.md` | ~200 | ~50 | Comprehensive documentation update |
| `tests/test_cli.py` | ~50 | - | Tests for callers command |

---

## Approval

- [ ] User has reviewed this plan
- [ ] All pre-resolved decisions approved
- [ ] Ready to proceed with implementation

**Sign-off:**

```
User: ________________  Date: ________
```
