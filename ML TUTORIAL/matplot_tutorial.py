import matplotlib.pyplot as plt 
import numpy as np



x = np.array([1,2,3,4])
y = np.array([1,4,7,4])
# z = np.array([12,46,17,4])

x1 = np.array([1,2,3,4])
y1 = np.array([1,4,7,4])



# plt.plot(x,marker = '+',ms=20,mec='r',color='y')
# plt.plot(y,marker = '+',ms=20,mec='r',ls='--',c='#4CAF50')
# plt.plot(z,marker = '+',ms=20,mec='r',ls='--',color = 'g')

# plt.title(' TUTURIAL GRAPH ')
# plt.xlabel(' X TESTING LABEL')
# plt.ylabel(' Y TESTING LABEL')
# plt.grid(color = 'blue',linestyle='--',linewidth=2)


# x = np.array([1,2,3,4])
# y = np.array([1,4,7,4])
# # z = np.array([12,46,17,4])

# x1 = np.array([1,2,3,4])
# y1 = np.array([5,4,11,4])


# plt.subplot(1,2,1)
# plt.plot(x,y)
# plt.title('PLOT 1 TITLE')


# plt.subplot(1,2,2)
# plt.plot(x1,y1)
# plt.title('PLOT 2 TITLE')

# plt.suptitle(' TUTORIAL GRAPH ')


# x = np.random.randint(100,size=(100))
# y = np.random.randint(100,size=(100))
# colors = np.random.randint(100,size=(100))

# plt.scatter(x,y,c=colors,cmap='nipy_spectral')

# plt.colorbar()
# x = [1,2,3,4,5,6]
#  x/sum(x)


x = [4,7,9,2]
mylabel = ['4','7','9','2']
myexplode = [0,0,0.2,0]

plt.pie(x,labels=mylabel,explode=myexplode,shadow=True,)
plt.legend()
plt.show()







classification_data = {
    'Age': [21,24,26,29,31,34,36,39,41,44,46,49,51,54,57],
    'Salary': [28000,33000,36000,41000,44000,47000,51000,57000,59000,62000,64000,67000,69000,71000,74000],
    'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'EducationLevel': ['HighSchool','Diploma','Diploma','BSc','BSc','MSc','MSc','PhD','PhD','BSc','MSc','HighSchool','Diploma','PhD','MSc'],
    'City': ['Lagos','Abuja','Kano','Lagos','Enugu','Portharcourt','Lagos','Abuja','Kano','Lagos','Enugu','Portharcourt','Lagos','Abuja','Kano'],
    'Department': ['IT','HR','Finance','IT','HR','Finance','IT','HR','Finance','IT','HR','Finance','IT','HR','Finance'],
    'Gender': ['Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male'],
    'Target': ['Yes','No','Yes','Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes']
}


regression_data = {
    'Age': [22,25,27,30,32,35,37,40,42,45,47,50,52,55,58],
    'Salary': [30000,35000,37000,42000,45000,48000,52000,58000,60000,63000,65000,68000,70000,72000,75000],
    'YearsExperience': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'EducationLevel': ['HighSchool','Diploma','Diploma','BSc','BSc','MSc','MSc','PhD','PhD','BSc','MSc','HighSchool','Diploma','PhD','MSc'],
    'City': ['Lagos','Abuja','Kano','Lagos','Enugu','Portharcourt','Lagos','Abuja','Kano','Lagos','Enugu','Portharcourt','Lagos','Abuja','Kano'],
    'Department': ['IT','HR','Finance','IT','HR','Finance','IT','HR','Finance','IT','HR','Finance','IT','HR','Finance'],
    'Gender': ['Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male','Female','Male'],
    'Target': [31000,36000,38000,43000,46000,49000,53000,59000,60500,64000,66000,69000,71000,73000,76000]
}
