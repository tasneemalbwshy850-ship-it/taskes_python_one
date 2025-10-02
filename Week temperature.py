import matplotlib.pyplot as plt

days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
temperatures = [30,33,31,28,25,28,30]

plt.plot(days, temperatures, marker="o", linestyle="-", color="black")
plt.title("Week temperature")
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.show()