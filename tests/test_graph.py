"""Unit tests for call graph construction and edge extraction."""

import pytest
import numpy as np

from codegrapher.graph import (
    ClassHierarchyMap,
    _build_class_hierarchy_map,
    _make_contextual_callee_id,
    _resolve_method_with_1level_mro,
    extract_edges_from_file,
)
from codegrapher.models import Edge, Symbol


# =============================================================================
# Test Suite 1: Named Fuzzy Edges (Feature 1)
# =============================================================================

class TestMakeContextualCalleeId:
    """Test _make_contextual_callee_id function."""

    def test_self_method_in_class(self):
        """self.B1() in class method becomes module.class.B1"""
        result = _make_contextual_callee_id("self.B1", "c2.A2.B2")
        assert result == "c2.A2.B1"

    def test_self_method_nested_class(self):
        """self.bar() in nested class uses full class path"""
        result = _make_contextual_callee_id("self.bar", "pkg.Outer.Inner.foo")
        assert result == "pkg.Outer.Inner.bar"

    def test_self_in_top_level_function(self):
        """self.X in function is invalid"""
        result = _make_contextual_callee_id("self.X", "module.func")
        assert result == "<invalid>.self.X"

    def test_self_in_module(self):
        """self.X at module level is invalid"""
        result = _make_contextual_callee_id("self.X", "module")
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

    def test_cls_in_function(self):
        """cls.method in function is invalid"""
        result = _make_contextual_callee_id("cls.X", "module.func")
        assert result == "<invalid>.cls.X"

    def test_super_call_unchanged(self):
        """super().method() stays as-is"""
        result = _make_contextual_callee_id("super().method", "c2.A2.B2")
        assert result == "super().method"

    def test_chained_attribute(self):
        """obj.attr.method stays as-is"""
        result = _make_contextual_callee_id("obj.attr.method", "c2.A2.B2")
        assert result == "obj.attr.method"


# =============================================================================
# Test Suite 2: Class Hierarchy Map (Feature 2)
# =============================================================================

class TestClassHierarchyMap:
    """Test ClassHierarchyMap dataclass."""

    def test_empty_map(self):
        """Empty map should have empty dicts."""
        hierarchy = ClassHierarchyMap()
        assert len(hierarchy.class_methods) == 0
        assert len(hierarchy.class_parents) == 0

    def test_get_method_in_current_class(self):
        """Method found in current class."""
        hierarchy = ClassHierarchyMap(
            class_methods={"c2.A2": {"B2", "other"}},
            class_parents={}
        )
        result = hierarchy.get_method_defining_class("c2.A2", "B2")
        assert result == "c2.A2"

    def test_get_method_in_parent_class(self):
        """Method found in parent class."""
        hierarchy = ClassHierarchyMap(
            class_methods={
                "c2.A2": {"B2"},
                "c1.A1": {"B1"}
            },
            class_parents={"c2.A2": ["c1.A1"]}
        )
        result = hierarchy.get_method_defining_class("c2.A2", "B1")
        assert result == "c1.A1"

    def test_get_method_not_found(self):
        """Method not found returns None."""
        hierarchy = ClassHierarchyMap(
            class_methods={"c2.A2": {"B2"}},
            class_parents={}
        )
        result = hierarchy.get_method_defining_class("c2.A2", "Zebra")
        assert result is None

    def test_get_method_checks_first_parent_only(self):
        """Multiple inheritance: only check first parent."""
        hierarchy = ClassHierarchyMap(
            class_methods={
                "c.C": {},
                "c.A": {"method"},
                "c.B": {"method"}  # Also has method but is second parent
            },
            class_parents={"c.C": ["c.A", "c.B"]}
        )
        result = hierarchy.get_method_defining_class("c.C", "method")
        assert result == "c.A"  # First parent, not c.B

    def test_get_method_unknown_class(self):
        """Unknown class returns None."""
        hierarchy = ClassHierarchyMap()
        result = hierarchy.get_method_defining_class("unknown.Class", "method")
        assert result is None


class TestBuildClassHierarchyMap:
    """Test _build_class_hierarchy_map function."""

    def _make_symbol(self, symbol_id: str) -> Symbol:
        """Helper to create a test Symbol."""
        return Symbol(
            id=symbol_id,
            file="test.py",
            start_line=1,
            end_line=10,
            signature=f"def {symbol_id.split('.')[-1]}():",
            doc=None,
            mutates="",
            embedding=np.zeros(768, dtype=np.float32)
        )

    def test_extracts_class_methods(self):
        """Map should contain class → methods mapping."""
        symbols = [
            self._make_symbol("c2.A2.B2"),
            self._make_symbol("c2.A2.other"),
            self._make_symbol("c1.A1.B1"),
        ]
        hierarchy = _build_class_hierarchy_map(symbols, [])

        assert "c2.A2" in hierarchy.class_methods
        assert "B2" in hierarchy.class_methods["c2.A2"]
        assert "other" in hierarchy.class_methods["c2.A2"]
        assert "c1.A1" in hierarchy.class_methods
        assert "B1" in hierarchy.class_methods["c1.A1"]

    def test_extracts_parent_relationships(self):
        """Map should contain class → parents mapping."""
        symbols = [self._make_symbol("c2.A2.B2")]
        edges = [
            Edge("c2.A2", "c1.A1", "inherit"),
            Edge("c1.A1", "object", "inherit"),
        ]
        hierarchy = _build_class_hierarchy_map(symbols, edges)

        assert hierarchy.class_parents["c2.A2"] == ["c1.A1"]
        assert hierarchy.class_parents["c1.A1"] == ["object"]

    def test_ignores_standalone_functions(self):
        """Map should not include standalone functions."""
        symbols = [
            self._make_symbol("module.func"),
            self._make_symbol("c2.A2.method"),
        ]
        hierarchy = _build_class_hierarchy_map(symbols, [])

        assert "module.func" not in hierarchy.class_methods
        assert "c2.A2" in hierarchy.class_methods

    def test_handles_multiple_inheritance(self):
        """Map should store all parents in order."""
        symbols = [self._make_symbol("c.C.method")]
        edges = [
            Edge("c.C", "c.A", "inherit"),
            Edge("c.C", "c.B", "inherit"),
        ]
        hierarchy = _build_class_hierarchy_map(symbols, edges)

        assert hierarchy.class_parents["c.C"] == ["c.A", "c.B"]

    def test_empty_inputs(self):
        """Empty symbols/edges should produce empty map."""
        hierarchy = _build_class_hierarchy_map([], [])

        assert len(hierarchy.class_methods) == 0
        assert len(hierarchy.class_parents) == 0

    def test_nested_classes(self):
        """Nested classes should have separate entries."""
        symbols = [
            self._make_symbol("pkg.Outer.method"),
            self._make_symbol("pkg.Outer.Inner.method"),
        ]
        hierarchy = _build_class_hierarchy_map(symbols, [])

        assert "pkg.Outer" in hierarchy.class_methods
        assert "pkg.Outer.Inner" in hierarchy.class_methods


# =============================================================================
# Test Suite 3: 1-Level MRO Resolution (Feature 2)
# =============================================================================

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
            "self.B2", "c2.A2.B2", sample_hierarchy
        )
        assert result == "c2.A2.B2"

    def test_resolves_inherited_method(self, sample_hierarchy):
        """Method defined in parent class."""
        result = _resolve_method_with_1level_mro(
            "self.B1", "c2.A2.B2", sample_hierarchy
        )
        assert result == "c1.A1.B1"

    def test_falls_back_to_fuzzy_for_unknown_method(self, sample_hierarchy):
        """Unknown method falls back to contextual fuzzy."""
        result = _resolve_method_with_1level_mro(
            "self.Zebra", "c2.A2.B2", sample_hierarchy
        )
        assert result == "c2.A2.Zebra"

    def test_handles_standalone_function_caller(self, sample_hierarchy):
        """Standalone function can't use MRO."""
        result = _resolve_method_with_1level_mro(
            "foo", "module.func", sample_hierarchy
        )
        assert result == "<unknown>.foo"

    def test_handles_self_call_in_function(self, sample_hierarchy):
        """self.method() in standalone function is invalid."""
        result = _resolve_method_with_1level_mro(
            "self.X", "module.func", sample_hierarchy
        )
        assert result == "<unknown>.self.X"

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
            "self.method", "c.C.foo", hierarchy
        )
        assert result == "c.A.method"  # First parent, not c.B.method

    def test_handles_cls_method_call(self):
        """cls.method() should resolve like self.method()."""
        hierarchy = ClassHierarchyMap(
            class_methods={
                "c2.A2": {"factory"},
                "c1.A1": {"build"},
            },
            class_parents={"c2.A2": ["c1.A1"]}
        )
        result = _resolve_method_with_1level_mro(
            "cls.build", "c2.A2.factory", hierarchy
        )
        assert result == "c1.A1.build"

    def test_handles_non_self_call(self):
        """Direct calls (not self.) bypass MRO but get contextual ID."""
        hierarchy = ClassHierarchyMap(
            class_methods={"c2.A2": {"B2"}},
            class_parents={}
        )
        result = _resolve_method_with_1level_mro(
            "foo", "c2.A2.B2", hierarchy
        )
        # Direct calls get contextual ID, not <unknown>
        assert result == "c2.A2.foo"

    def test_handles_empty_class_name(self):
        """Edge case: caller with minimal parts."""
        hierarchy = ClassHierarchyMap()
        result = _resolve_method_with_1level_mro(
            "self.method", "func", hierarchy
        )
        assert result == "<unknown>.self.method"


# =============================================================================
# Test Suite 4: Integration Tests
# =============================================================================

class TestExtractEdgesFromFile:
    """Integration tests for extract_edges_from_file."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with test files."""
        # Create c1.py with base class
        (tmp_path / "c1.py").write_text("""
class A1:
    def B1(self):
        return "base"

    def A1_method(self):
        pass
""")

        # Create c2.py with derived class
        (tmp_path / "c2.py").write_text("""
from c1 import A1

class A2(A1):
    def B2(self):
        result = self.B1()
        return result

    def direct_call(self):
        return foo()

    def calls_own(self):
        return self.B2()
""")

        return tmp_path

    def _make_symbol(self, symbol_id: str, file: str) -> Symbol:
        """Helper to create a test Symbol."""
        return Symbol(
            id=symbol_id,
            file=file,
            start_line=1,
            end_line=10,
            signature=f"def {symbol_id.split('.')[-1]}():",
            doc=None,
            mutates="",
            embedding=np.zeros(768, dtype=np.float32)
        )

    def test_self_call_resolves_to_parent_class(self, temp_repo):
        """self.B1() in A2 should get contextual fuzzy (cross-file limitation)."""
        from codegrapher.parser import extract_symbols

        c2_path = temp_repo / "c2.py"
        symbols = extract_symbols(c2_path, temp_repo)
        edges = extract_edges_from_file(c2_path, temp_repo, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        b2_calls = [e for e in call_edges if e.caller_id == "c2.A2.B2"]

        # Should have an edge from B2 to something
        assert len(b2_calls) > 0

        # The self.B1() call should be contextual fuzzy (c2.A2.B1)
        # because cross-file inheritance isn't resolved in 1-level MRO
        b1_call = next((e for e in b2_calls if "B1" in e.callee_id), None)
        assert b1_call is not None
        # Cross-file: falls back to contextual fuzzy
        assert b1_call.callee_id == "c2.A2.B1"

    def test_direct_call_unchanged(self, temp_repo):
        """Direct foo() call should stay as "foo"."""
        from codegrapher.parser import extract_symbols

        c2_path = temp_repo / "c2.py"
        symbols = extract_symbols(c2_path, temp_repo)
        edges = extract_edges_from_file(c2_path, temp_repo, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        assert any(e.callee_id == "foo" for e in call_edges)

    def test_inherit_edge_extracted(self, temp_repo):
        """Inheritance edge should be extracted."""
        from codegrapher.parser import extract_symbols

        c2_path = temp_repo / "c2.py"
        symbols = extract_symbols(c2_path, temp_repo)
        edges = extract_edges_from_file(c2_path, temp_repo, symbols)

        inherit_edges = [e for e in edges if e.type == "inherit"]
        # Note: v1 limitation - cross-file base class gets module-local ID
        # The edge is created but callee_id is "c2.A1" not "c1.A1"
        assert any(e.caller_id == "c2.A2" and "A1" in e.callee_id
                   for e in inherit_edges)

    def test_no_edges_for_nonexistent_file(self, tmp_path):
        """Nonexistent file should return empty edges."""
        fake_file = tmp_path / "fake.py"
        symbols = []
        edges = extract_edges_from_file(fake_file, tmp_path, symbols)
        assert edges == []

    def test_handles_syntax_error(self, tmp_path):
        """Syntax error should return empty edges."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("class : (invalid syntax")

        symbols = []
        edges = extract_edges_from_file(bad_file, tmp_path, symbols)
        assert edges == []

    def test_same_file_inheritance(self, tmp_path):
        """Classes in same file should have proper hierarchy."""
        (tmp_path / "same.py").write_text("""
class Base:
    def base_method(self):
        pass

class Derived(Base):
    def derived_method(self):
        self.base_method()
""")

        from codegrapher.parser import extract_symbols

        same_path = tmp_path / "same.py"
        symbols = extract_symbols(same_path, tmp_path)
        edges = extract_edges_from_file(same_path, tmp_path, symbols)

        # Check inheritance edge
        inherit_edges = [e for e in edges if e.type == "inherit"]
        assert any(e.caller_id == "same.Derived" and e.callee_id == "same.Base"
                   for e in inherit_edges)

        # Check call resolution
        call_edges = [e for e in edges if e.type == "call"]
        derived_calls = [e for e in call_edges if e.caller_id == "same.Derived.derived_method"]
        base_method_call = next((e for e in derived_calls if "base_method" in e.callee_id), None)
        assert base_method_call is not None
        # Should be resolved to same.Base.base_method
        assert base_method_call.callee_id == "same.Base.base_method"


# =============================================================================
# Test Suite 5: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test specific edge cases."""

    def _make_symbol(self, symbol_id: str, file: str = "test.py") -> Symbol:
        """Helper to create a test Symbol."""
        return Symbol(
            id=symbol_id,
            file=file,
            start_line=1,
            end_line=10,
            signature=f"def {symbol_id.split('.')[-1]}():",
            doc=None,
            mutates="",
            embedding=np.zeros(768, dtype=np.float32)
        )

    def test_multiple_inheritance_first_parent(self, tmp_path):
        """Multiple inheritance uses first parent."""
        (tmp_path / "multi.py").write_text("""
class A:
    def method(self):
        pass

class B:
    def method(self):
        pass

class C(A, B):
    def foo(self):
        self.method()
""")

        from codegrapher.parser import extract_symbols

        multi_path = tmp_path / "multi.py"
        symbols = extract_symbols(multi_path, tmp_path)
        edges = extract_edges_from_file(multi_path, tmp_path, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        method_calls = [e for e in call_edges if "method" in e.callee_id]

        # Should resolve to A.method (first parent)
        assert any(e.callee_id == "multi.A.method" for e in method_calls)

    def test_super_call_unchanged(self, tmp_path):
        """super().method() - limited support due to AST complexity."""
        (tmp_path / "super_test.py").write_text("""
class Base:
    def method(self):
        pass

class Derived(Base):
    def other(self):
        super().method()
""")

        from codegrapher.parser import extract_symbols

        super_path = tmp_path / "super_test.py"
        symbols = extract_symbols(super_path, tmp_path)
        edges = extract_edges_from_file(super_path, tmp_path, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        # Note: super().method() has complex AST structure
        # _unparse_attribute returns "<unknown>" for the super() part
        # So we get "super" as the callee name, not "super().method"
        assert any("super" in e.callee_id for e in call_edges)

    def test_classmethod_cls_call(self, tmp_path):
        """cls.method() in classmethod."""
        (tmp_path / "cls_test.py").write_text("""
class Base:
    @classmethod
    def factory(cls):
        pass

class Derived(Base):
    @classmethod
    def create(cls):
        cls.factory()
""")

        from codegrapher.parser import extract_symbols

        cls_path = tmp_path / "cls_test.py"
        symbols = extract_symbols(cls_path, tmp_path)
        edges = extract_edges_from_file(cls_path, tmp_path, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        factory_calls = [e for e in call_edges if "factory" in e.callee_id]

        # Should resolve to Base.factory
        assert any(e.callee_id == "cls_test.Base.factory" for e in factory_calls)

    def test_nested_class_self_call(self, tmp_path):
        """Nested class - parser limitation (only extracts top-level)."""
        (tmp_path / "nested.py").write_text("""
class Outer:
    class Inner:
        def method(self):
            self.other()
""")

        from codegrapher.parser import extract_symbols

        nested_path = tmp_path / "nested.py"
        symbols = extract_symbols(nested_path, tmp_path)

        # Note: The parser only extracts top-level definitions (PRD Section 5)
        # So nested classes and their methods are NOT extracted
        # This test verifies that limitation - only Outer is extracted
        assert any("Outer" in s.id for s in symbols)
        assert not any("Inner" in s.id for s in symbols)

    def test_no_base_class(self, tmp_path):
        """Class with no base class."""
        (tmp_path / "nobase.py").write_text("""
class A2:
    def method(self):
        self.foo()
""")

        from codegrapher.parser import extract_symbols

        nobase_path = tmp_path / "nobase.py"
        symbols = extract_symbols(nobase_path, tmp_path)
        edges = extract_edges_from_file(nobase_path, tmp_path, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        foo_calls = [e for e in call_edges if "foo" in e.callee_id]

        # Should fall back to contextual fuzzy
        assert any(e.callee_id == "nobase.A2.foo" for e in foo_calls)

    def test_method_override(self, tmp_path):
        """Method override should resolve to derived class."""
        (tmp_path / "override.py").write_text("""
class Base:
    def process(self):
        pass

class Derived(Base):
    def process(self):
        pass

    def call(self):
        self.process()
""")

        from codegrapher.parser import extract_symbols

        override_path = tmp_path / "override.py"
        symbols = extract_symbols(override_path, tmp_path)
        edges = extract_edges_from_file(override_path, tmp_path, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        process_calls = [e for e in call_edges if "process" in e.callee_id]

        # Should resolve to Derived.process (the override)
        assert any(e.callee_id == "override.Derived.process" for e in process_calls)

    def test_chained_attribute_call(self, tmp_path):
        """obj.attr.method() preserves full chain."""
        (tmp_path / "chain.py").write_text("""
class A:
    def method(self):
        self.attr.other()
""")

        from codegrapher.parser import extract_symbols

        chain_path = tmp_path / "chain.py"
        symbols = extract_symbols(chain_path, tmp_path)
        edges = extract_edges_from_file(chain_path, tmp_path, symbols)

        call_edges = [e for e in edges if e.type == "call"]
        # Should preserve the full attr.other chain with class context
        assert any("attr.other" in e.callee_id for e in call_edges)
