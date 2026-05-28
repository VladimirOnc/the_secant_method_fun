import numpy as np
import matplotlib.pyplot as plt


class SideException(Exception):
    """
    Это исключение используется для сигнализации о недопустимых или непредвиденных побочных условиях при
    работе с нелинейными функциями в этом модуле.
    """
    pass


class NoLinerFunction:
    def f(self, x):
        """Вычисляет значение нелинейной функции, определенной для данного класса.
        """
        return -(x**3) + 12 * np.sin(3 * x) - 5 * x

    def __call__(self, x):
        """Позволяет вызывать экземпляр класса как функцию."""
        return self.f(x)

    def secant_method(self, x0, x1, acceptable=1e-8, max_iterations=50):
        """
        Метод секущих для нахождения корня.
        Параметры:
        x0, x1 — начальные приближения
        tolerance — допустимая погрешность
        max_iterations — максимальное число итераций
        """
        """Проверка начальных приближений"""
        f0 = self.f(x0)
        f1 = self.f(x1)

        print(f"Начальные приближения: x0={x0}, x1={x1}")
        print(f"f(x0)={f0}, f(x1)={f1}")

        if f0 * f1 > 0:
            print("ВНИМАНИЕ: функция не меняет знак на [x0, x1]. Сходимость не гарантирована.")
        elif f0 * f1 == 0:
            if f0 == 0:
                print(f"x0 является корнем: f({x0}) = 0")
                return x0
            else:
                print(f"x1 является корнем: f({x1}) = 0")
                return x1

        
        for _ in range(max_iterations):
            f0 = self.f(x0)  # Вызов метода f() с аргументом x0
            f1 = self.f(x1)  # Вызов метода f() с аргументом x1

            if abs(f1 - f0) < 1e-12:  # Защита от деления на ноль
                print("Деление на ноль в методе секущих")
                break

            # Формула метода секущих
            x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
            print(f"Итерация {_+1}: x={x_next:.6f}, f(x)={self.f(x_next):.6f}")
            if abs(x_next - x1) < acceptable:
                print(f"Сходимость достигнута на итерации {_+1}")
                return x_next

            x0, x1 = x1, x_next

        print("Достигнуто максимальное число итераций")
        return x1
    
    # Использование
if __name__ == "__main__":
    func = NoLinerFunction()
    arg1 = np.double(input("Введите начальное приближение x0: "))
    arg2 = np.double(input("Введите начальное приближение x1: "))
    root = func.secant_method(arg1, arg2)
    print(f"Корень: {root}")
    
    # Построение графика функции
    # x = np.linspace(-5, 5, 100) # задаём диапазон значений x , 100 точек от -5 до 5# 100 точек от -5 до 5
    # y = -x**3 + 12 * np.sin(3 * x) - 5 * x   # вычисляем значения y
    # plt.figure(figsize=(10, 6))
    # plt.gcf().canvas.manager.set_window_title("Метод секущих") 
    # plt.plot(x, y, label='$f(x) = -x^3 + 12\\sin(3x) - 5x$', color='red')   # строим график
    # plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    # plt.xlabel("x") # подписываем ось x
    # plt.ylabel("f(x)") # подписываем ось y
    # plt.title("График функции $f(x) = -x^3 + 12 \sin(3x) - 5x$") # добавляем заголовок
    # plt.grid(True) # добавляем сетку
    # plt.legend()  
    # plt.show() 