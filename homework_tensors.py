import torch


def task_1_1():
    print("--- 1.1 Создание тензоров ---")
    # Тензор размером 3x4 со случайными числами от 0 до 1
    t_rand = torch.rand(3, 4)
    print("Случайный тензор 3x4:\n", t_rand)

    # Тензор размером 2x3x4, заполненный нулями
    t_zeros = torch.zeros(2, 3, 4)
    print("\nТензор из нулей 2x3x4:\n", t_zeros)

    # Тензор размером 5x5, заполненный единицами
    t_ones = torch.ones(5, 5)
    print("\nТензор из единиц 5x5:\n", t_ones)

    # Тензор размером 4x4 с числами от 0 до 15
    t_range = torch.arange(16).reshape(4, 4)
    print("\nТензор 4x4 от 0 до 15:\n", t_range)


def task_1_2():
    print("\n--- 1.2 Операции с тензорами ---")
    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    # Транспонирование тензора A
    A_T = A.T  # или A.transpose(0, 1)
    print("Транспонированный A (размер):", A_T.shape)

    # Матричное умножение A и B
    matmul_res = torch.matmul(A, B)  # или A @ B
    print("Матричное умножение A x B:\n", matmul_res)

    # Поэлементное умножение A и транспонированного B
    # B.T имеет размерность (3, 4), совпадающую с A (3, 4)
    elem_mult = A * B.T
    print("\nПоэлементное умножение A * B.T:\n", elem_mult)

    # Сумма всех элементов тензора A
    sum_A = A.sum()
    print("\nСумма элементов A:", sum_A.item())


def task_1_3():
    print("\n--- 1.3 Индексация и срезы ---")
    # Создаем тензор 5x5x5 для демонстрации
    tensor_5x5x5 = torch.arange(125).reshape(5, 5, 5)

    # Первая строка (первый вектор по первой двумерной матрице/срезу)
    first_row = tensor_5x5x5[0, 0, :]
    print("Первая строка [0, 0, :]:\n", first_row)

    # Последний столбец
    last_col = tensor_5x5x5[:, :, -1]
    print("\nПоследний столбец [:, :, -1]:\n", last_col)

    # Подматрица размером 2x2 из центра тензора
    # Для размера 5x5 центральные элементы располагаются по индексам 1:3 или 2:4
    submatrix_center = tensor_5x5x5[2, 1:3, 1:3]
    print("\nЦентральная подматрица 2x2:\n", submatrix_center)

    # Все элементы с четными индексами
    even_indexed = tensor_5x5x5[::2, ::2, ::2]
    print("\nЭлементы с четными индексами (шаг 2):\n", even_indexed)


def task_1_4():
    print("\n--- 1.4 Работа с формами ---")
    t_24 = torch.arange(24)

    print("Форма 2x12:\n", t_24.reshape(2, 12))
    print("\nФорма 3x8:\n", t_24.reshape(3, 8))
    print("\nФорма 4x6:\n", t_24.reshape(4, 6))
    print("\nФорма 2x3x4:\n", t_24.reshape(2, 3, 4))
    print("\nФорма 2x2x2x3:\n", t_24.reshape(2, 2, 2, 3))


if __name__ == "__main__":
    task_1_1()
    task_1_2()
    task_1_3()
    task_1_4()