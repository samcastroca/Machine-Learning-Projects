import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = len(x)
        self.w = 0
        self.b = 0

    def initialize_parameters(self):
        self.w = np.random.randn()
        self.b = 0

    def gradients(self):
        dw = np.sum((self.w * self.x + self.b - y)*self.x)/self.m
        db = np.sum(self.w * self.x + self.b - y)/self.m

        return dw, db


    def fit(self, num_iterations, learning_rate = 0.01):
        self.initialize_parameters()
        for i in range(num_iterations):
            dw, db = self.gradients()
            self.w = self.w - learning_rate*dw
            self.b = self.b - learning_rate*db
            if (i+1)%10 == 0:
                print(self.mse())

    def mse(self):
        return (1/(2*self.m))*np.sum((self.w * self.x + self.b - y)**2)
    
pendiente = 2
intercepto = 1
num_puntos = 100
desviacion_estandar_ruido = 5

# Generar los valores de x
x = np.linspace(0, 10, num_puntos)

# Generar los valores de y siguiendo la línea recta y = mx + b con algo de ruido
y = pendiente * x + intercepto + np.random.normal(0, desviacion_estandar_ruido, num_puntos)

lr = LinearRegression(x,y)
lr.fit(10)
x1 = np.linspace(0, 10, 100)

# Calcular los valores de y para la línea recta y = mx + b
y2 = lr.w * x + lr.b

# Crear la gráfica
plt.plot(x1, y2, label=f'y = {pendiente}x + {intercepto}', color='red')

plt.scatter(x,y)
plt.show()