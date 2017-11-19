import csv


def change_quality(file, sep=','):
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=sep)
        rows = []
        for row in reader:
            if int(row[11]) > 5:
                row[11] = 1
            else:
                row[11] = 0
            rows.append(row)
    with open("winequality-red_cleaned.csv", 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(rows)


change_quality("winequality-red.csv")
