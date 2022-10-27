import pandas as pd
mark_sort = pd.read_csv('std_marks.csv')
print(mark_sort)
print(f"\nSorting Data (Name):\n{mark_sort.sort_values(['Name'])}")
print(f"\nSorting Data (Marks):\n{mark_sort.sort_values(['Marks'])}")
