import csv

# Пример матрицы
matrix = [
    [(1.1, 2.2), (3.3, 4.4)],
    [(5.5, 6.6), (7.7, 8.8)]
]

# Имя выходного файла
output_file = "matrix.csv"

# Запись матрицы в CSV
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter=",")
    for row in matrix:
        # Преобразуем каждую ячейку в формат "a;b"
        formatted_row = [";".join(map(str, cell)) for cell in row]
        writer.writerow(formatted_row)

print(f"Матрица сохранена в файл {output_file}.")
