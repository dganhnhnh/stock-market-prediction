
import matplotlib.pyplot as plt
import helper

# # 2. Train model

import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


class Individual:
    def __init__(self, X_train, Y_train, position=[0, 0, 0], velocity=[0, 0, 0], cognitive_coef=1, social_coef=1):
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.position = position
        self.velocity = velocity
        self.model: SVR = SVR(
            kernel='sigmoid', gamma=self.position[0], coef0=self.position[1], epsilon=self.position[2], C=150)
        self.best_position = position
        self.X_train = X_train
        self.Y_train = Y_train
        self.fitness = self.compute_fitness(self.model)

    def compute_fitness(self, model: SVR):
        model.fit(self.X_train, self.Y_train)
        return -mean_squared_error(self.Y_train, model.predict(self.X_train))

    def update_indi_best(self, new_position):
        new_model: SVR = SVR(
            kernel='sigmoid', gamma=new_position[0], coef0=new_position[1], epsilon=new_position[2])
        new_model_fitness = self.compute_fitness(new_model)
        if (new_model_fitness >= self.fitness):
            self.best_position = new_position
            self.model = new_model
        return self.best_position

    def update_velocity(self, best_pos_in_pop):
        for i in range(len(self.position)):
            r = np.random.uniform(0, 1, 2)
            self.velocity[i] = self.velocity[i] + self.cognitive_coef*r[0]*(
                self.best_position[i] - self.position[i]) + self.social_coef*r[1]*(best_pos_in_pop[i] - self.position[i])
        return self.velocity

    def update_position(self):
        new_pos = np.zeros(len(self.position))
        for i in range(len(self.position)):
            if (i == 1):
                new_pos[i] = self.velocity[i] + self.position[i]
            else:
                new_pos[i] = abs(self.velocity[i] + self.position[i])
        self.update_indi_best(new_pos)
        self.position = new_pos
        return new_pos


class Population:
    def __init__(self, X_train, Y_train, pop_size, cognitive_coef, social_coef):
        self.pop_size = pop_size
        gamma = np.random.uniform(0, 0.01, pop_size)
        coef0 = np.random.uniform(-0.1, 0.1, pop_size)
        epsilon = np.random.uniform(0, 0.1, pop_size)
        self.pop = []
        for i in range(pop_size):
            indi = Individual(X_train, Y_train, position=[gamma[i], coef0[i], epsilon[i]],
                              cognitive_coef=cognitive_coef, social_coef=social_coef)
            self.pop.append(indi)
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.X_train=X_train
        self.Y_train=Y_train

    def new_Individual(self, individual: Individual) -> Individual:
        return Individual(self.X_train, self.Y_train, individual.position, individual.velocity, individual.cognitive_coef, individual.social_coef)

    def get_best_individual(self) -> Individual:
        self.pop = sorted(self.pop, key=lambda x: x.fitness)
        best_individual: Individual = self.new_Individual(self.pop[-1])
        return best_individual

    def update_population(self):
        best_particle: Individual = self.get_best_individual()
        for i in range(self.pop_size):
            temp = self.new_Individual(self.pop[i])
            temp.update_velocity(best_particle.position)
            temp.update_position()
            temp.compute_fitness(temp.model)
            if (temp.fitness > self.pop[i].fitness):
                self.pop[i] = temp
            if (temp.fitness > best_particle.fitness):
                best_particle = temp

        return


class PSO_SVR_Model:
    def __init__(self):
        self.name = 'PSO-SVR'
        self.input_shape = 8
        self.population_size = 100
        self.cognitive_coef = 1
        self.social_coef = 1
        self.number_of_generation = 10

    def prepare_data(self, ticker, train_pct=0.7):
        self.ticker = ticker
        data = helper.prepare_data(ticker, self.input_shape, train_pct, 'minmax')
        self.X_train = data['X_train']
        self.Y_train = data['Y_train']
        self.X_test = data['X_test']
        self.Y_test = data['Y_test']

    def train(self):
        self.population = Population(
            self.X_train, self.Y_train, self.population_size, self.cognitive_coef, self.social_coef)
        fitness = []

        for i in range(self.number_of_generation):
            self.population.update_population()
            best_individual = self.population.get_best_individual()
            fitness.append(best_individual.fitness)
            percentage = i/self.number_of_generation*100
            print(f'PSO: {percentage}%')

        self.population.get_best_individual().fitness
        self.model = self.population.get_best_individual().model
        self.Y_pred = self.model.predict(self.X_test)
        return self.Y_pred
        
    def calculate_loss(self):
        mse, r2, rmse, mape, mae = helper.calculate_loss(self.model, self.X_test, self.Y_test)
        return mse, r2, rmse, mape, mae


    def plot_result(self):
        plt.plot(self.Y_pred, c='red', label='Prediction')
        plt.plot(self.Y_test, c='blue', label='True value')
        plt.legend(loc='upper right')
        plt.title(f'{self.ticker} - {self.name}')
        plt.show()
