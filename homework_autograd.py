"""
Задание 2: Автоматическое дифференцирование
"""

import torch


def task_2_1():
    print("--- 2.1 Простые вычисления с градиентами ---")
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)

    # f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y*z
    f = x**2 + y**2 + z**2 + 2 * x * y * z
    f.backward()

    print(f"Вычисленное значение f: {f.item()}")
    print(f"Градиент df/dx: {x.grad.item()}")
    print(f"Градиент df/dy: {y.grad.item()}")
    print(f"Градиент df/dz: {z.grad.item()}")

    # Аналитическая проверка:
    # df/dx = 2*x + 2*y*z = 2*(2) + 2*(3)*(4) = 4 + 24 = 28
    # df/dy = 2*y + 2*x*z = 2*(3) + 2*(2)*(4) = 6 + 16 = 22
    # df/dz = 2*z + 2*x*y = 2*(4) + 2*(2)*(3) = 8 + 12 = 20
    grad_x_analyt = 2 * x.item() + 2 * y.item() * z.item()
    grad_y_analyt = 2 * y.item() + 2 * x.item() * z.item()
    grad_z_analyt = 2 * z.item() + 2 * x.item() * y.item()

    print("\nАналитические градиенты:")
    print(
        f"df/dx = {grad_x_analyt}, df/dy = {grad_y_analyt}, df/dz = {grad_z_analyt}"
    )


def task_2_2():
    print("\n--- 2.2 Градиент функции потерь (MSE) ---")
    # Исходные данные
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])  # Ожидаем w=2, b=1

    # Параметры модели
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Предсказание и расчёт MSE
    y_pred = w * x + b
    mse_loss = torch.mean((y_pred - y_true) ** 2)

    mse_loss.backward()

    print(f"MSE Loss: {mse_loss.item()}")
    print(f"Градиент dMSE/dw: {w.grad.item()}")
    print(f"Градиент dMSE/db: {b.grad.item()}")


def task_2_3():
    print("\n--- 2.3 Цепное правило ---")
    x = torch.tensor(1.5, requires_grad=True)

    # Составная функция f(x) = sin(x^2 + 1)
    f = torch.sin(x**2 + 1)

    # Способ 1: Использование autograd.grad
    (grad_autograd,) = torch.autograd.grad(f, x, create_graph=True)

    # Способ 2: Обычный backward
    f.backward()

    print(f"Значение f(x): {f.item()}")
    print(f"Градиент через autograd.grad: {grad_autograd.item()}")
    print(f"Градиент через backward(): {x.grad.item()}")

    # Аналитическая проверка: df/dx = cos(x^2 + 1) * 2*x
    analyt_grad = torch.cos(x**2 + 1) * 2 * x
    print(f"Аналитический градиент: {analyt_grad.item()}")


if __name__ == "__main__":
    task_2_1()
    task_2_2()
    task_2_3()