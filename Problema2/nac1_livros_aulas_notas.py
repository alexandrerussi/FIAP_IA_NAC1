from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

arq = open("FIAP - IA - NAC1 - dados - livros_freq_nota.txt", "r")
lines = arq.readlines()
arq.close()

qtd_livros = []
freq_aulas = []
nota_final = []

for line in lines:
    line_values = line.split()
    qtd_livros.append(float(line_values[0]))
    freq_aulas.append(float(line_values[1]))
    nota_final.append(float(line_values[2]))

entrada = []
for i in range(0, 40):
    entrada.append([qtd_livros[i], freq_aulas[i]])

linear_reg = linear_model.LinearRegression()
linear_reg.fit(entrada, nota_final)

a = linear_reg.coef_
b = linear_reg.intercept_

z_predict = linear_reg.predict(entrada)

nota_2l_11aulas = linear_reg.predict([[2.0, 11.0]])
print("Nota do aluno que leu 2 livros e assistiu 11 aulas: ", nota_2l_11aulas)

nota_0l_5aulas = linear_reg.predict([[0.0, 5.0]])
print("Nota do aluno que leu 0 livros e assistiu 5 aulas: ", nota_0l_5aulas)

nota_4l_20aulas = linear_reg.predict([[4.0, 20.0]])
print("Nota do aluno que leu 4 livros e assistiu 20 aulas: ", nota_4l_20aulas)

nota_2l_10aulas = linear_reg.predict([[2.0, 10.0]])
print("Nota do aluno que leu 2 livros e assistiu 10 aulas: ", nota_2l_10aulas)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(qtd_livros, freq_aulas, nota_final)
ax.plot_trisurf(qtd_livros, freq_aulas, z_predict, linewidth=0, antialiased=False, color='red', alpha='0.5')
ax.set_xlabel('Livros')
ax.set_ylabel('Freq Aulas')
ax.set_zlabel('Nota')
# plt.title("Livros lidos e Aulas assistidas x Notas finais")
plt.savefig("livros_aulas_notas.png")
plt.show()
