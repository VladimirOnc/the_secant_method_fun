import pytest
import sys
sys.path.append('C:/Users/shama/Desktop/the_secant_method_fun')

from app.NoLinerFunction import SideException

class TestNoLinerFunction:
    
    @pytest.mark.parametrize(
        "message, expected_message",
        [
            # пустое сообщение строки
            pytest.param(
                "",
                "",
                id="",
            ),
            # простое текстовое сообщение
            pytest.param(
                "simple error",
                "simple error",
                id="",
            ),
            # сообщение русском языке
            pytest.param(
                "нелинейная функция вышла за пределы области определения",
                "нелинейная функция вышла за пределы области определения",
                id="",
            ),
            # Длинное сообщение
            pytest.param(
                "x" * 1000,
                "x" * 1000,
                id="",
            ),
            # числовое сообщение, преобразованное в строку по Exception
            pytest.param(
                123,
                "123",
                id="",
            ),
            # набор значений
            pytest.param(
                ("a", "b"),
                "('a', 'b')",
                id="",
            ),
        ],
    )
    def test_side_exception_message_representation(self, message, expected_message):
        exc = SideException(message)
        # Исключение должно быть экземпляром SideException и базовым типом исключения
        assert isinstance(exc, SideException)
        assert isinstance(exc, Exception)

        # Args должен содержать исходное сообщение (согласно поведению Exception)
        assert exc.args == (message,)

        # Представление строки должно соответствовать ожидаемой форме строки
        assert str(exc) == expected_message

    @pytest.mark.parametrize(
        "message",
        [
            # Сообщения, похожие на ошибку (текстуально имитирующие различные условия ошибки)
            pytest.param(
                "division by zero during secant method iteration",
                id="",
            ),
            pytest.param(
                "maximum iterations exceeded in nonlinear solver",
                id="",
            ),
            pytest.param(
                "function not defined in given interval",
                id="",
            ),
            pytest.param(
                "derivative approximated as zero, cannot proceed",
                id="",
            ),
        ],
    )
    def test_side_exception_raised_and_caught(self,message):
        with pytest.raises(SideException) as exc_info:
            raise SideException(message)
        assert str(exc_info.value) == message
        assert isinstance(exc_info.value, SideException)
        assert isinstance(exc_info.value, Exception)


    def test_side_exception_without_arguments_behaviour(self) -> None:
        exc = SideException()
        # Никакие правила, переданные не должны отражать поведение Exception
        assert exc.args == ()
        assert not str(exc)
        assert repr(exc).startswith("SideException()")
        
if __name__=='__main__':
    pytest.main()
