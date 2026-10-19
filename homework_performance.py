"""
Задание 3: Сравнение производительности CPU vs CUDA
"""

import time
import torch


def measure_time_cpu(func, *args, num_runs=5):
    """Измерение времени выполнения операции на CPU."""
    start = time.time()
    for _ in range(num_runs):
        func(*args)
    end = time.time()
    return ((end - start) / num_runs) * 1000  # Перевод в миллисекунды


def measure_time_gpu(func, *args, num_runs=5):
    """Измерение времени выполнения операции на GPU с синхронизацией CUDA."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Прогрев GPU
    func(*args)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(num_runs):
        func(*args)
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs


def compare_performance():
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu_available = torch.cuda.is_available()

    print(f"GPU доступен: {is_gpu_available}")
    if is_gpu_available:
        print(f"Устройство: {torch.cuda.get_device_name(0)}")

    # Размеры тензоров
    shape = (128, 512, 512)

    # 3.1 Подготовка данных
    A_cpu = torch.randn(shape)
    B_cpu = torch.randn(shape)

    if is_gpu_available:
        A_gpu = A_cpu.to(device_gpu)
        B_gpu = B_cpu.to(device_gpu)

    operations = [
        ("Матричное умножение", lambda x, y: torch.matmul(x, y)),
        ("Поэлементное сложение", lambda x, y: x + y),
        ("Поэлементное умножение", lambda x, y: x * y),
        ("Транспонирование", lambda x, y: x.transpose(-1, -2)),
        ("Сумма элементов", lambda x, y: x.sum()),
    ]

    print("\n" + "=" * 65)
    print(
        f"{'Операция':<25} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}"
    )
    print("=" * 65)

    for name, op in operations:
        time_cpu = measure_time_cpu(op, A_cpu, B_cpu)

        if is_gpu_available:
            time_gpu = measure_time_gpu(op, A_gpu, B_gpu)
            speedup = time_cpu / time_gpu if time_gpu > 0 else 0
            speedup_str = f"{speedup:.2f}x"
            gpu_str = f"{time_gpu:.2f}"
        else:
            gpu_str = "N/A"
            speedup_str = "N/A"

        print(
            f"{name:<25} | {time_cpu:<10.2f} | {gpu_str:<10} | {speedup_str:<10}"
        )
    print("=" * 65)


def print_analysis():
    print("\n--- 3.4 Анализ результатов ---")
    print(
        """
1. Какие операции получают наибольшее ускорение на GPU?
   - Наибольшее ускорение получают тяжелые параллельные вычисления, такие как матричное умножение (matmul) и трехмерные/тензорные свертки, где тысячи ядер GPU могут одновременно выполнять параллельные операции.

2. Почему некоторые операции могут быть медленнее на GPU?
   - Операции с малым объемом данных или требующие последовательной обработки могут работаться медленнее из-за накладных расходов на запуск ядер CUDA (overhead) и задержек синхронизации.

3. Как размер матриц влияет на ускорение?
   - Чем больше размерность матриц, тем эффективнее параллельная архитектура GPU утилизирует свои вычислительные блоки. На маленьких матрицах преобладают накладные расходы.

4. Что происходит при передаче данных между CPU и GPU?
   - Передача данных происходят по шине PCIe, которая обладает ограниченной пропускной способностью. Часто копирование данных с CPU на GPU занимает больше времени, чем само вычисление, поэтому важно минимизировать пересылки данных в цикле обучения.
"""
    )


if __name__ == "__main__":
    compare_performance()
    print_analysis()