import csv


def change_quality(file, sep=','):
    """
    change the data in the quality column to fit 2 clases logistic regression


    :param file:
    :param sep:

    """

    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=sep)
        rows = []
        for row in reader:
            if int(row[11]) > 5:
                row[11] = 1
            else:
                row[11] = 0
            rows.append(row)
    with open("%s_cleaned.csv" % file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(rows)


change_quality("winequality-red.csv")
