import pandas as pd
mark_sort = pd.read_csv('std_marks.csv')
print(mark_sort)
print(f"\nSorting Data (Name):\n{file.sort_values(['Name'])}")
print(f"\nSorting Data (Marks):\n{file.sort_values(['Marks'])}")
