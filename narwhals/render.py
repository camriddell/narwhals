from __future__ import annotations

from dataclasses import dataclass, field, replace
from inspect import getclosurevars
from itertools import chain
from typing import Any, Callable

import narwhals as nw
from narwhals._utils import flatten
from narwhals.expr_cat import ExprCatNamespace
from narwhals.expr_dt import ExprDateTimeNamespace
from narwhals.expr_list import ExprListNamespace
from narwhals.expr_name import ExprNameNamespace
from narwhals.expr_str import ExprStringNamespace
from narwhals.expr_struct import ExprStructNamespace
from narwhals.functions import When


def replace_render(replacement):
    """Renderer that completely overrides the default output."""
    return lambda fname, parent, kwargs: replacement


def unary_render(fname, parent, kwargs):
    """Renderer for unary functions"""
    rendered_args = (
        Node.from_expr(a).render() if isinstance(a, nw.Expr) else repr(a)
        for a in kwargs.values()
    )
    if parent is None:
        return f"{fname}({', '.join(rendered_args)})"
    return f"{parent.render()}.{fname}({', '.join(rendered_args)})"


def binary_render(symbol, right_kw="other"):
    """Renderer for binary functions; must specify the calling function keyword for the 'other' argument"""

    def _render(fname, left, kwargs):
        right = kwargs.pop("other")
        if isinstance(right, Node):
            right = right.render()
        else:
            right = repr(right)
        return f"({left.render()} {symbol} {right})"

    return _render


def multioutput_render(expr_varname="flat_exprs"):
    """Renderer to flatten the call signature of expressions that have multioutput"""

    def _render(fname, parent, kwargs):
        rendered_args = (
            value.render() if isinstance(value, Node) else repr(value)
            for value in kwargs[expr_varname]
        )

        if parent is None:
            return f"{fname}({', '.join(rendered_args)})"
        return f"{parent.render()}.{fname}({', '.join(rendered_args)})"

    return _render


def filtered_render(include=[], **exclude):
    """Renderer to include/conditionally filter arguments from call signatures"""
    _default = object()

    def _render(fname, parent, kwargs):
        to_include = chain(
            include,
            [
                k
                for k in kwargs
                if exclude.get(k, kwargs[k]) != kwargs[k] and k not in include
            ],
        )
        rendered_args = (
            v.render() if isinstance(v := kwargs[key], Node) else repr(v)
            for key in to_include
        )

        if parent is None:
            return f"{fname}({', '.join(rendered_args)})"
        return f"{parent.render()}.{fname}({', '.join(rendered_args)})"

    return _render


def nary_render(fname, parent, kwargs):
    """Renderer to display all arguments of a passed in function

    note: at the time of writing, this is the default renderer.
    """
    rendered_args = (
        Node.from_expr(a).render() if isinstance(a, nw.Expr) else repr(a)
        for a in flatten([*kwargs.values()])
    )
    if parent is None:
        return f"{fname}({', '.join(rendered_args)})"
    return f"{parent.render()}.{fname}({', '.join(rendered_args)})"


def whenthen_render(fname, parent, kwargs):
    """Renderer specifically for when/then combinations

    note: this is tightly coupled renderer (see the checks for "When" and isinstance(..., When)
    """
    rendered_args = (
        Node.from_expr(a).render() if isinstance(a, nw.Expr) else repr(a)
        for a in flatten([*kwargs.values()])
    )
    parent = replace(parent, func=nw.when)
    return f"{parent.render()}.{fname}({', '.join(rendered_args)})"


def no_args_render(fname, parent, kwargs):
    """Renderer that strips all arguments"""
    if parent is None:
        return f"{fname}()"
    return f"{parent.render()}.{fname}()"


def fill_null_render(fname, parent, kwargs):
    """Renderer for specifically handling the modalities of `full_null`"""
    if (value := kwargs["value"]) is not None:
        to_render = {"value": value}
    elif (strategy := kwargs["strategy"]) is not None:
        to_render = {"strategy": strategy}
        if (limit := kwargs["limit"]) is not None:
            to_render["limit"] = limit
    kwarg_string = ", ".join(f"{k}={v!r}" for k, v in to_render.items())

    if parent is None:
        return f"{fname}({kwarg_string})"
    return f"{parent.render()}.{fname}({kwarg_string})"


@dataclass(frozen=True)
class Node:
    expr: nw.Expr = field(repr=False)
    func: Callable
    kwargs: dict[str, Any] = field(default_factory=dict)
    parent: Node | None = field(default=None, repr=False)
    accessor: str | None = None

    renderers = {
        "__add__": binary_render("+"),
        "__sub__": binary_render("-"),
        "__mul__": binary_render("*"),
        "__floordiv__": binary_render("//"),
        "__truediv__": binary_render("/"),
        "__pow__": binary_render("**"),
        "__lt__": binary_render("<"),
        "__le__": binary_render("<="),
        "__gt__": binary_render(">"),
        "__ge__": binary_render(">="),
        "__eq__": binary_render("=="),
        "__and__": binary_render("&"),
        "__or__": binary_render("|"),
        # '__xor__': '^',
        # '__neg__': '.negate()',
        "col": multioutput_render("flat_names"),
        "all_": replace_render("all()"),
        "len_": replace_render("len()"),
        "lit": filtered_render(include=["value"], dtype=None),
        "all_horizontal": multioutput_render("flat_exprs"),
        "any_horizontal": multioutput_render("flat_exprs"),
        "concat_str": multioutput_render("flat_exprs"),
        "when": multioutput_render("flat_exprs"),
        "then": whenthen_render,
        "otherwise": filtered_render(include=["value"]),
        "quantile": no_args_render,
        "cum_count": no_args_render,
        "cum_min": no_args_render,
        "cum_max": no_args_render,
        "cum_prod": no_args_render,
        "cum_sum": no_args_render,
        # 'is_duplicated': replace_render('is_duplicated()'), # needs an actual stack to recover entry
        "rolling_mean": no_args_render,
        "rolling_std": no_args_render,
        "rolling_sum": no_args_render,
        "rolling_var": no_args_render,
        "var": no_args_render,
        "is_in": unary_render,
        "replace_strict": filtered_render(include=["old", "new"]),
        "fill_null": fill_null_render,
        "is_between": filtered_render(include=["lower_bound", "upper_bound"]),
        "log": no_args_render,
        "std": no_args_render,
        "over": filtered_render(include=["flat_partition_by"], flat_order_by=[]),
        "rank": no_args_render,
        "ewm_mean": no_args_render,
    }

    accessor_map = {
        ExprCatNamespace: "cat",
        ExprDateTimeNamespace: "dt",
        ExprListNamespace: "list",
        ExprNameNamespace: "name",
        ExprStringNamespace: "str",
        ExprStructNamespace: "struct",
    }

    def __post_init__(self):
        # parse the function name out of its qualified name
        qualname_parts = self.func.__qualname__.split(".")
        if qualname_parts[0] in {"Expr", "When", "Then"}:
            fname = qualname_parts[1]
        elif qualname_parts[0].endswith("Namespace"):
            if self.accessor is None:
                raise valueError("that should not have happened")
            fname = f"{self.accessor}.{qualname_parts[1]}"
        else:
            fname = qualname_parts[0]
        object.__setattr__(self, "fname", fname)

    @classmethod
    def from_expr(cls, expr: nw.Expr) -> Node:
        func = getclosurevars(expr._to_compliant_expr).nonlocals["to_compliant_expr"]
        kwargs = getclosurevars(func).nonlocals

        accessor = None
        try:
            parent_expr = kwargs.pop("self")
        except KeyError:
            parent = None
        else:
            # recursively create all parent Nodes from enclosed expressions
            is_namespace = isinstance(
                parent_expr,
                (
                    ExprCatNamespace,
                    ExprDateTimeNamespace,
                    ExprListNamespace,
                    ExprNameNamespace,
                    ExprStringNamespace,
                    ExprStructNamespace,
                ),
            )

            if is_namespace:
                accessor = cls.accessor_map[type(parent_expr)]
                parent_expr = parent_expr._expr
            elif isinstance(parent_expr, When):
                parent_expr = parent_expr._predicate

            parent = cls.from_expr(parent_expr)

        return cls(func=func, kwargs=kwargs, expr=expr, parent=parent, accessor=accessor)

    def parents(self):
        """Traverses the nodes, yields the current Node first."""
        while self is not None:
            yield self
            self = self.parent

    def render(self):
        """Represent the Node as a string. Akin to a __repr__ of the Expression"""
        kwargs = {}
        for k, v in self.kwargs.items():
            if isinstance(v, nw.Expr):
                kwargs[k] = type(self).from_expr(v)
            else:
                kwargs[k] = v

        render_func = self.renderers.get(self.fname, nary_render)
        return render_func(self.fname, self.parent, kwargs)


def render(expr: nw.Expr):
    return Node.from_expr(expr).render()
