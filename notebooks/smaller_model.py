import random
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt

from collections import Counter

class Agent:

    def __init__(self, race, religion, income) :
        self.race = race       # Random distribution of 5
        self.religion = religion    # Dependent
        # self.type = type    # Ruler, elite, or person
        self.income = income

        self.revolutionary = 'placeholder'

    def get_attribute(self, attribute):
        if (attribute == "race"):
            return self.race
        elif (attribute == "religion"):
            return self.religion
        else:
            return 0
        
        
    def __str__(self) -> str:
        return f'Race: {self.race}, Religion: {self.religion}, Income: {self.income}'
        

class Environment:

    def __init__(self, population_size, empty_ratio, race_count, religion_count):
        self.race_count = race_count
        self.religion_count = religion_count

        self.population_size = int(np.sqrt(population_size))**2 
        self.dim = int(np.sqrt(self.population_size))

        population = self.generate_population(self.population_size, empty_ratio, race_count, religion_count)
        self.people = [person for person in population if type(person) == Agent]

        self.population_grid = np.reshape(population, (int(np.sqrt(self.population_size)), int(np.sqrt(self.population_size))))

        # Within a tenth percentile of the max income
        self.similarity_threshold = 0.3
        self.income_similarity_threshold = 1000 / 10
        self.generate_demographics()
        self.initialize_population_capital()
        self.initialize_policy()

        # self.population = np.reshape(population, (int(np.sqrt(self.population_size)), int(np.sqrt(self.population_size))))

    
    def generate_population(self, population_size, empty_ratio, race_count, religion_count):

        p = [1-empty_ratio, empty_ratio]
        choice = np.random.choice([0,1], size=population_size, p=p)
        population = [self.create_agent(race_count, religion_count) if i == 0 else -1 for i in choice]
        return population
    
    
    def scheilling(self):
        dim = len(self.population_grid)
        
        for (row, col), value in np.ndenumerate(self.population_grid):
            person = self.population_grid[row, col]

            if type(person) is Agent:
                rlb = max(0, row-1)
                rub = min(dim, row+2)
                clb = max(0, col-1)
                cub = min(dim, col+2)

                neighborhood = self.population_grid[rlb:rub, clb:cub]
                neighborhood_size = np.size(neighborhood)
                n_empty_houses = len(np.where(neighborhood == -1)[0])
                if neighborhood_size != n_empty_houses + 1:
                    # similarity calculations
                    n_similar = self.similarity_ratio(person, neighborhood)
                    similarity_ratio = n_similar / (neighborhood_size - n_empty_houses - 1.)
                    is_unhappy = (similarity_ratio < self.similarity_threshold)
                    if is_unhappy:
                        empty_houses = list(zip(np.where(self.population_grid == -1)[0], np.where(self.population_grid == -1)[1]))
                        random_house = random.choice(empty_houses)
                        self.population_grid[random_house] = person
                        self.population_grid[row,col] = -1
     

    def similarity_ratio(self, person, neighborhood):
        race_similarity = 0
        religion_similarity = 0
        income_similarity = 0
        for row in neighborhood:
            for neighbor in row:
                if type(neighbor) == Agent:
                    if neighbor.race == person.race: race_similarity += 1
                    if neighbor.religion == person.religion: religion_similarity += 1
                    if neighbor.income > person.income: self.income_similarity_threshold += 1
            
        return (race_similarity * 0.33) + (religion_similarity * 0.33) + (income_similarity * 0.33) 
                


    
    def initialize_population_capital(self):
        # print(self.race_income)
        # print(self.race_demographics)
        self.race_capital = self.race_income + self.race_demographics

        self.religion_capital = self.religion_income + self.religion_demographics

        # does this div work? 
        self.race_capital /= self.race_capital.sum()
        self.religion_capital /= self.religion_capital.sum()

    

    def initialize_policy(self):
        # np.full creates 2D array, need to index into 0
        self.race_policy = np.full((1,self.race_count),0.5)[0]
        self.religion_policy = np.full((1,self.religion_count),0.5)[0]


    def enact_policy(self):
        for person in self.people:
            benefit = np.average([self.race_policy[person.race], self.religion_policy[person.religion]])
            p = [1-benefit, benefit]

            choice = np.random.choice([-5,5], size=1, p=p)
            person.income += choice

            race = person.race
            self.race_income[race] += choice


    def generate_demographics(self):
        race_demographics = Counter([person.race for person in self.people])
        religion_demographics = Counter([person.religion for person in self.people])
        self.race_demographics = np.array(list(race_demographics.values()), dtype=np.float64) 
        self.race_demographics /= self.race_demographics.sum()
        self.religion_demographics = np.array(list(religion_demographics.values()), dtype=np.float64)
        self.religion_demographics /= self.religion_demographics.sum()

        self.race_income = np.zeros(self.race_count)
        self.religion_income = np.zeros(self.religion_count)

        for person in self.people: 
            race = person.race
            religion = person.religion
            income = person.income
            self.race_income[race] += income
            self.religion_income[religion] += income


    def create_agent(self, race_count, religion_count):
        race = np.random.randint(race_count)
        # look up normal popualtion age distribution
        # age = int(abs(np.random.normal(0.5, 0.3)) * 100)
        income = int(abs(np.random.normal(0.5, 0.3)) * 1000)
        religion = np.random.randint(religion_count)

        return Agent(race, religion, income)
    
    def get_plot_data(self):
        x = []
        y = []
        race_data = []
        religion_data = []
        for i in range(len(self.population_grid)):
            for j in range(len(self.population_grid[i])):
                x.append(i)
                y.append(j)
                if self.population_grid[i][j] == -1:
                    race_data.append(-1)
                    religion_data.append(-1)
                else:
                    race_data.append(self.population_grid[i][j].race)
                    religion_data.append(self.population_grid[i][j].race)

        return x,y,race_data,religion_data
        
    

class Govern:
    def __init__(self, government_size, government_type, budget, revenue):
        self.government_size = government_size
        self.government_type = government_type
        # Assume that debt exists, but the 0 limit of a budget represents fiscal strain
        self.budget = budget
        self.revenue = revenue

    def load_environment(self, environment):
        self.environment = environment

    def create_policy(self, cost):
        self.budget -= cost
        



        
        



pop_size = 2500
empty_ratio = 0.2
race_count = 3
religion_count = 5

sim1 = Govern(100, "democracy", 1000, 1000)

env1 = Environment(pop_size,empty_ratio, race_count, religion_count)
sim1.load_environment(env1)





# ----- Visualization -----


# def get_data(population_grid):
#     x = []
#     y = []
#     race_data = []
#     for i in range(len(population_grid)):
#         for j in range(len(population_grid[i])):
#             x.append(i)
#             y.append(j)
#             if population_grid[i][j] == -1:
#                 race_data.append(-1)
#             else:
#                 race_data.append(population_grid[i][j].race)
    
#     race_data = np.array(race_data)
#     col = np.where(race_data == -1, 'w', np.where(race_data<1, 'b','r'))

#     return x,y,col

# initial_x, initial_y, initial_col = get_data(env.population_grid)

# plt.figure(figsize=(4,4))

# # plt.subplot(121)
# plt.axis('off')
# plt.scatter(initial_x,initial_y,c=initial_col,marker='s',linewidth=0, s=100)

# st.title("Schelling's Model of Segregation")

# populatio_plot = st.pyplot(plt)

# progress_bar = st.progress(0)

# n_iterations = st.sidebar.number_input("Number of iterations", 10)

# if st.sidebar.button('Run Simulation'):

#     for i in range(n_iterations):
#         env.scheilling()

#         new_x, new_y, new_col = get_data(env.population_grid)

#         plt.figure(figsize=(4,4))
#         plt.axis('off')
#         plt.scatter(new_x,new_y,c=new_col,marker='s',linewidth=0, s=100)

#         populatio_plot.pyplot(plt)
#         plt.close('all')

#         progress_bar.progress((i+1.)/n_iterations)

        
