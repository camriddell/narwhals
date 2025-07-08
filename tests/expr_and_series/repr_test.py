from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.render import render


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.col("a", "b"), "col('a', 'b')"),
        (
            nw.exclude("a"),
            "exclude('a')",
        ),  # internally uses frozenset, ordering is not preserved
        (nw.nth(1, 2), "nth(1, 2)"),
        (nw.all(), "all()"),
        (nw.len(), "len()"),
        (nw.sum("a", "b"), "col('a', 'b').sum()"),
        (nw.mean("a", "b"), "col('a', 'b').mean()"),
        (nw.median("a", "b"), "col('a', 'b').median()"),
        (nw.min("a", "b"), "col('a', 'b').min()"),
        (nw.max("a", "b"), "col('a', 'b').max()"),
        (nw.sum_horizontal("a", "b"), "sum_horizontal('a', 'b')"),
        (nw.min_horizontal("a", "b"), "min_horizontal('a', 'b')"),
        (nw.max_horizontal("a", "b"), "max_horizontal('a', 'b')"),
        (nw.all_horizontal("a", "b", ignore_nulls=True), "all_horizontal('a', 'b')"),
        (nw.any_horizontal("a", "b", ignore_nulls=False), "any_horizontal('a', 'b')"),
        (nw.lit(1), "lit(1)"),
        (nw.lit(1, dtype=nw.Int64()), "lit(1, Int64)"),
        (
            nw.coalesce("a", "b", nw.lit(1, dtype=nw.Int64())),
            "coalesce('a', 'b', lit(1, Int64))",
        ),
        (nw.concat_str("a", "b", "c", separator="-"), "concat_str('a', 'b', 'c')"),
    ],
)
def test_repr_functions(expr: nw.Expr, expected: str):
    assert render(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.when("a").then("b"), "when('a').then('b')"),
        (nw.when("a").then("b").otherwise(1), "when('a').then('b').otherwise(1)"),
    ],
)
def test_repr_whenthen(expr: nw.Expr, expected: str):
    assert render(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.col("a").mean(), "col('a').mean()"),
        (nw.col("a").sum(), "col('a').sum()"),
        (nw.col("a") + nw.col("b"), "(col('a') + col('b'))"),
        (nw.col("a") + nw.col("b"), "(col('a') + col('b'))"),
        (nw.col("a") / nw.col("b"), "(col('a') / col('b'))"),
        (nw.col("a") // nw.col("b"), "(col('a') // col('b'))"),
        (nw.col("a").abs(), "col('a').abs()"),
        (nw.col("a").alias("b"), "col('a').alias('b')"),
        (nw.col("a").all(), "col('a').all()"),
        (nw.col("a").any(), "col('a').any()"),
        (nw.col("a").arg_max(), "col('a').arg_max()"),
        (nw.col("a").arg_min(), "col('a').arg_min()"),
        (nw.col("a").cast(nw.Float64()), "col('a').cast(Float64)"),
        (nw.col("a").clip(0, 1), "col('a').clip(0, 1)"),
        (nw.col("a").count(), "col('a').count()"),
        (nw.col("a").cum_count(), "col('a').cum_count()"),
        (nw.col("a").cum_max(), "col('a').cum_max()"),
        (nw.col("a").cum_min(), "col('a').cum_min()"),
        (nw.col("a").cum_prod(), "col('a').cum_prod()"),
        (nw.col("a").cum_sum(), "col('a').cum_sum()"),
        (nw.col("a").diff(), "col('a').diff()"),
        (nw.col("a").drop_nulls(), "col('a').drop_nulls()"),
        (nw.col("a").ewm_mean(), "col('a').ewm_mean()"),
        (nw.col("a").exp(), "col('a').exp()"),
        (nw.col("a").fill_null(value=1), "col('a').fill_null(value=1)"),
        (
            nw.col("a").fill_null(strategy="forward"),
            "col('a').fill_null(strategy='forward')",
        ),
        (
            nw.col("a").fill_null(strategy="forward", limit=10),
            "col('a').fill_null(strategy='forward', limit=10)",
        ),
        (nw.col("a").filter(nw.col("a") > 0), "col('a').filter((col('a') > 0))"),
        (nw.col("a").is_between(0, 1), "col('a').is_between(0, 1)"),
        # (nw.col("a").is_duplicated(), "col('a').is_duplicated()"), # requires explicit stack to recover entry
        (nw.col("a").is_finite(), "col('a').is_finite()"),
        (nw.col("a").is_first_distinct(), "col('a').is_first_distinct()"),
        (nw.col("a").is_in([1, 2, 3]), "col('a').is_in([1, 2, 3])"),
        (nw.col("a").is_last_distinct(), "col('a').is_last_distinct()"),
        (nw.col("a").is_nan(), "col('a').is_nan()"),
        (nw.col("a").is_null(), "col('a').is_null()"),
        (nw.col("a").is_unique(), "col('a').is_unique()"),
        (nw.col("a").kurtosis(), "col('a').kurtosis()"),
        (nw.col("a").len(), "col('a').len()"),
        (nw.col("a").log(), "col('a').log()"),
        (nw.col("a").max(), "col('a').max()"),
        (nw.col("a").mean(), "col('a').mean()"),
        (nw.col("a").median(), "col('a').median()"),
        (nw.col("a").min(), "col('a').min()"),
        (nw.col("a").mode(), "col('a').mode()"),
        (nw.col("a").n_unique(), "col('a').n_unique()"),
        (nw.col("a").null_count(), "col('a').null_count()"),
        (nw.col("a").sum().over("b"), "col('a').sum().over(['b'])"),
        (nw.col("a").sum().over("b", "c"), "col('a').sum().over(['b', 'c'])"),
        (nw.col("a").quantile(0.5, "nearest"), "col('a').quantile()"),
        (nw.col("a").rank(), "col('a').rank()"),
        (nw.col("a").replace_strict("x", "y"), "col('a').replace_strict('x', 'y')"),
        (nw.col("a").rolling_mean(3), "col('a').rolling_mean()"),
        (nw.col("a").rolling_std(3), "col('a').rolling_std()"),
        (nw.col("a").rolling_sum(3), "col('a').rolling_sum()"),
        (nw.col("a").rolling_var(3), "col('a').rolling_var()"),
        (nw.col("a").round(2), "col('a').round(2)"),
        (nw.col("a").shift(1), "col('a').shift(1)"),
        (nw.col("a").skew(), "col('a').skew()"),
        (nw.col("a").sqrt(), "col('a').sqrt()"),
        (nw.col("a").std(), "col('a').std()"),
        (nw.col("a").sum(), "col('a').sum()"),
        (nw.col("a").unique(), "col('a').unique()"),
        (nw.col("a").var(), "col('a').var()"),
    ],
)
def test_repr_methods(expr: nw.Expr, expected: str):
    assert render(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.col("a").str.to_uppercase(), "col('a').str.to_uppercase()"),
        (nw.col("a").str.len_chars(), "col('a').str.len_chars()"),
        (nw.col("a").list.len(), "col('a').list.len()"),
        (nw.col("a").dt.year(), "col('a').dt.year()"),
        (nw.col("a").dt.hour(), "col('a').dt.hour()"),
    ],
)
def test_repr_namespace(expr: nw.Expr, expected: str):
    assert render(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            nw.col("a", "b") ** 2 + nw.col("c").mean(),
            "((col('a', 'b') ** 2) + col('c').mean())",
        ),
        (
            nw.col("a").dt.hour() * 3600 + nw.col("a").dt.minute() * 60,
            "((col('a').dt.hour() * 3600) + (col('a').dt.minute() * 60))",
        ),
        (
            (nw.col("a") + nw.col("b")).mean().round(2),
            "(col('a') + col('b')).mean().round(2)",
        ),
        (
            (nw.col("a") ** 2 + nw.col("b") ** 0.5).max().cast(nw.Float64()),
            "((col('a') ** 2) + (col('b') ** 0.5)).max().cast(Float64)",
        ),
        (
            nw.col("a").filter(nw.col("a") > 0).sum(),
            "col('a').filter((col('a') > 0)).sum()",
        ),
        (
            (nw.col("a") + nw.col("b") * nw.col("c")).log(),
            "(col('a') + (col('b') * col('c'))).log()",
        ),
        (nw.col("a").dt.minute().sum(), "col('a').dt.minute().sum()"),
        (nw.col("a").cat.get_categories(), "col('a').cat.get_categories()"),
    ],
)
def test_repr_complex(expr: nw.Expr, expected: str):
    assert render(expr) == expected
