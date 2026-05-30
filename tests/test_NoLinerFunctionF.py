import math
import sys
from typing import Any, Literal
import numpy as np
import pytest
sys.path.append("C:/Users/shama/Desktop/the_secant_method_fun")
from app.NoLinerFunction import NoLinerFunction


class TestNoLinerFunctionF:
    @pytest.mark.parametrize(
        "x, expected",
        [
            pytest.param(0.0, 0.0, id=""),
            pytest.param(1.0, -(1.0**3) + 12 * np.sin(3 * 1.0) - 5 * 1.0, id=""),
            pytest.param(
                -2.0,
                -((-2.0) ** 3) + 12 * np.sin(3 * -2.0) - 5 * -2.0,
                id="",
            ),
            pytest.param(
                1e-8,
                -(1e-8**3) + 12 * np.sin(3 * 1e-8) - 5 * 1e-8,
                id="",
            ),
            pytest.param(
                -1e-8,
                -((-1e-8) ** 3) + 12 * np.sin(3 * -1e-8) - 5 * -1e-8,
                id="",
            ),
            pytest.param(
                10.0,
                -(10.0**3) + 12 * np.sin(30.0) - 5 * 10.0,
                id="",
            ),
        ],
    )
    def test_f_computes_expected_value(self, x: float, expected: float | Any):
        func = NoLinerFunction()
        result = func.f(x)
        assert math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-12)


class TestNoLinerFunctionCall:
    @pytest.mark.parametrize(
        "x",
        [
            pytest.param(0.0, id=""),
            pytest.param(2.5, id=""),
            pytest.param(-3.3, id=""),
        ],
    )
    def test_call_delegates_to_f(self, x: float, monkeypatch: pytest.MonkeyPatch):
        func = NoLinerFunction()
        called = {"flag": False, "arg": None}

        def fake_f(arg):
            called["flag"] = True
            called["arg"] = arg
            return 42.0

        monkeypatch.setattr(func, "f", fake_f)
        result = func(x)
        assert called["flag"] is True
        assert called["arg"] == x
        assert result == 42


class TestSecantMethodInitialSignChecks:
    @pytest.mark.parametrize(
        "x0, x1, root_at, expected_root",
        [
            pytest.param(0, 1, "x0", 0, id=""),
            pytest.param(0, 1, "x1", 1, id=""),
        ],
    )
    def test_initial_exact_root_at_x0(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        x0, x1 = 0.0, 1.0
        func = NoLinerFunction()
        original_f = func.f

        def fake_f(x):
            if x == x0:
                return 0.0 if x == x0 else original_f(x)
        monkeypatch.setattr(func, "f", fake_f)
        root = func.secant_method(x0, x1)
        captured = capsys.readouterr().out
        assert math.isclose(root, x0, rel_tol=1e-12, abs_tol=1e-12)
        assert f"x0 является корнем: f({x0}) = 0" in captured

    def test_initial_exact_root_at_x0(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        x0, x1 = 0.0, 1.0
        func = NoLinerFunction()
        original_f = func.f

        def fake_f(x):
            return 0.0 if x == x0 else original_f(x)

        monkeypatch.setattr(func, "f", fake_f)
        root = func.secant_method(x0, x1)
        captured = capsys.readouterr().out
        assert math.isclose(root, x0, rel_tol=1e-12, abs_tol=1e-12)
        assert f"x0 является корнем: f({x0}) = 0" in captured
    
    @pytest.mark.parametrize("x0, x1, acceptable, max_iterations",
        [
            pytest.param(-1.0, 1.0, 1e-8, 50, id=""),
            pytest.param(-5.0, 5.0, 1e-6, 100, id=""),
            pytest.param(1.0, 3.0, 1e-5, 50, id=""),
        ],
    )
    def test_secant_method_converges_to_root(
        self, x0: float, x1: float, acceptable: float, max_iterations: Literal[50] | Literal[100], capsys: pytest.CaptureFixture[str]
    ):

        func = NoLinerFunction()
        root = func.secant_method(x0, x1, acceptable=acceptable, max_iterations=max_iterations)
        captured = capsys.readouterr().out
        assert isinstance(root, float)
        assert abs(func.f(root)) < 1e-4
        assert "Итерация" in captured
        assert "Сходимость достигнута" in captured or "Достигнуто максимальное число итераций" in captured

    def test_secant_method_hits_max_iterations(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ):
        func = NoLinerFunction()
        def slow_f(x):
            return 1.0
        monkeypatch.setattr(func, "f", slow_f)
        result = func.secant_method(0.0, 1.0, acceptable=1e-20, max_iterations=3)
        captured = capsys.readouterr().out
        assert isinstance(result, float)
        assert "Достигнуто максимальное число итераций" in captured

    def test_secant_method_division_by_zero_guard(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
        func = NoLinerFunction()
        call_count = {"count": 0}
        def f_with_same_value(_):
            call_count["count"] += 1
            return 2.0
        monkeypatch.setattr(func, "f", f_with_same_value)
        result = func.secant_method(0.0, 1.0, acceptable=1e-8, max_iterations=5)
        captured = capsys.readouterr().out
        assert "Деление на ноль в методе секущих" in captured
        assert "Достигнуто максимальное число итераций" in captured
        assert isinstance(result, float)
        assert call_count["count"] >= 2


class TestSecantMethodOutputAndSideEffects:
    def test_prints_initial_info_and_iterations(self, capsys: pytest.CaptureFixture[str]):
        func = NoLinerFunction()
        _ = func.secant_method(-1.0, 0.5, acceptable=1e-3, max_iterations=5)
        captured = capsys.readouterr().out
        assert "Начальные приближения:" in captured
        assert "f(x0)=" in captured
        assert "f(x1)=" in captured
        assert "Итерация 1:" in captured or "Сходимость достигнута" in captured
    def test_secant_method_return_is_last_x1_on_non_convergence(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
        func = NoLinerFunction()
        xs = []
        def track_x_values(x):
            xs.append(x)
            return x + 1.0
        monkeypatch.setattr(func, "f", track_x_values)
        result = func.secant_method(0.0, 2.0, acceptable=1e-20, max_iterations=2)
        captured = capsys.readouterr().out
        assert (
            "Достигнуто максимальное число итераций" in captured
            or "Сходимость достигнута" in captured
        )
        assert isinstance(result, float)
        assert len(xs) >= 2


if __name__ == "__main__":
    pytest.main()
    