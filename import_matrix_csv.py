import csv

output_file= "matrix.csv"

# Чтение матрицы из CSV
with open(output_file, mode="r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter=",")
    matrix = [
        [tuple(map(float, cell.split(";"))) for cell in row]
        for row in reader
    ]

print("Загруженная матрица:", matrix)
