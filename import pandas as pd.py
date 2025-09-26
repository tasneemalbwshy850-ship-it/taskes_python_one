import pandas as pd

students_data = [{"name": "Alice", "age": 24, "grade": "A","Marks":85},
                 {"name": "Bob", "age": 22, "grade": "B","Marks":78},
                 {"name": "Charlie", "age": 23, "grade": "C","Marks":92},
                 {"name": "David", "age": 21, "grade": "B","Marks":65},
                 {"name": "Eva", "age": 20, "grade": "A","Marks":74}
                 ]
df = pd.DataFrame(students_data)
print(df)

df.head(3)
print(df.head(3))

df[["name","Marks"]]
print(df[["name","Marks"]])

df.loc[df["grade"] == "A"]

print(df.loc[df["grade"] == "A"])
