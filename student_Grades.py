student_Grades ={
    "Rahma" : [90, 22, 50, 85],
    "Islam" : [50, 60, 50, 61],
    "Alex" : [90, 50, 70, 66],


}
for student, grades in student_Grades.items():
    average = sum(grades) / len(grades)
    print(f"{student} : {average}")

