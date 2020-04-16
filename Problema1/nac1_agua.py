from sklearn import linear_model
import matplotlib.pyplot as plt

# abrindo arquivo e salvando linhas na variável info
arq = open("FIAP - IA - NAC1 - dados - agua.txt", "r")
lines = arq.readlines()
arq.close()

temp_F = []
pressure = []

# salvando pressao e temperatura, convertendo para float cada valor
for line in lines:
    line_values = line.split()
    try:
        pressure.append(float(line_values[1]))
        temp_F.append([float(line_values[2])])
    except:
        print('Erro ao converter p/ float')

# plotando pontos sem regressão linear
plt.scatter(temp_F, pressure, color='blue')
plt.xlabel("Temperatura (F)")
plt.ylabel("Pressão (inHg)")
# plt.title("Ponto de ebulição da água em diferentes pressões barométricas")
plt.savefig("temperatura_pressao_noregression.png")

# regressão linear
linear_reg = linear_model.LinearRegression()
linear_reg.fit(temp_F, pressure)

# y = ax + b
a = linear_reg.coef_
b = linear_reg.intercept_

# plotando pontos sem regressão e reta da regressão linear
y_pressure_predict = linear_reg.predict(temp_F)
plt.scatter(temp_F, pressure, color='blue')
plt.plot(temp_F, y_pressure_predict, color='red')
plt.xlabel("Temperatura (F)")
plt.ylabel("Pressão (inHg)")
# plt.title("Regressão Linear: ponto de ebulição da água em \ndiferentes pressões barométricas")
plt.savefig("temperatura_pressao_regression.png")

# valor de pressão em 200F
pressure_200F = linear_reg.predict([[200.0]])
print("Valor de pressão para temperatura de 200 F:", pressure_200F)

pressure_215F = linear_reg.predict([[215.0]])
print("Valor de pressão para temperatura de 215 F:", pressure_215F)
