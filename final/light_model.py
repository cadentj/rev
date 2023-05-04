import random
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

from collections import Counter

class Agent:

    def __init__(self, race, religion, income) :
        self.race = race     
        self.religion = religion
        self.income = income

    def get_attribute(self, attribute):
        if (attribute == "race"):
            return self.race
        elif (attribute == "religion"):
            return self.religion
        else:
            return 0
        
        
    def __str__(self) -> str:
        return f'Race: {self.race}, Religion: {self.religion}, Income: {self.income}'
        
def norm(arr):
        return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

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
            
        return (race_similarity * 0.3) + (religion_similarity * 0.2) + (income_similarity * 0.5) 
                
    
    def initialize_population_capital(self):
        self.race_capital = (norm(self.race_income) * 0.7) + (norm(self.race_demographics) * 0.3)
        self.religion_capital = (norm(self.religion_income) * 0.7) + (norm(self.religion_demographics) * 0.3)

    def initialize_policy(self):
        # np.full creates 2D array, need to index into 0
        self.race_policy = np.full((1,self.race_count),0.5)[0]
        self.religion_policy = np.full((1,self.religion_count),0.5)[0]

    def step(self):
        # self.scheilling()
        self.enact_policy()

    def update_policy(self, new_race_policy, new_religion_policy):
        self.race_policy = new_race_policy
        self.religion_policy = new_religion_policy

    def enact_policy(self):
        for person in self.people:
            benefit = np.average([self.race_policy[person.race], self.religion_policy[person.religion]])
            p = [1-benefit, benefit]

            choice = np.random.choice([-5,5], size=1, p=p)
            person.income += choice
            
            # Update race_income demographic
            race = person.race
            self.race_income[race] += choice

    def generate_demographics(self):
        race_demographics = Counter([person.race for person in self.people])
        religion_demographics = Counter([person.religion for person in self.people])

        self.race_demographics = np.array(list(race_demographics.values()), dtype=np.float64) 
        self.religion_demographics = np.array(list(religion_demographics.values()), dtype=np.float64)

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
    def __init__(self, budget):
        # Assume that debt exists, but the 0 limit of a budget represents fiscal strain
        self.starting_budget = budget
        self.budget = budget

    def load_environment(self, environment):
        self.environment = environment

    def get_revolutionary_sentiment(self):
        race_sentiment = (norm(self.environment.race_income) * 0.5) + (norm(self.environment.race_capital) * 0.5)

        religion_sentiment = (norm(self.environment.religion_income) * 0.5) + (norm(self.environment.religion_capital) * 0.5)

        combined_sentiment = (np.average(race_sentiment) + np.average(religion_sentiment)) / 2
        self.average_sentiment = (combined_sentiment * 0.4) + (self.budget/self.starting_budget * 0.6)

        return self.average_sentiment

    def act(self):
        self.environment.step()

    def create_policy(self, cost, races, religions):
        self.budget -= cost
        if self.budget <= 0: return -1

        current_race_policy = self.environment.race_policy
        current_religion_policy = self.environment.religion_policy
        current_race_capital = self.environment.race_capital
        current_religion_capital = self.environment.religion_capital

        base_success = cost / self.budget
        change = np.random.choice([-0.05, 0.05], size=1, p=[1-base_success, base_success])

        if change > 0:
            for race in races: current_race_policy[race] += change
            for religion in religions: current_religion_policy[religion] += change
        else:
            for race in races: 
                does_defend = np.random.choice([False, True], size=1, p=[1-current_race_capital[race],current_race_capital[race]])
                if not does_defend : current_race_policy[race] += change
            for religion in religions:
                does_defend = np.random.choice([False, True], size=1, p=[1-current_religion_capital[religion],current_religion_capital[religion]])
                if not does_defend : current_race_policy[race] += change

        self.environment.update_policy(current_race_policy, current_religion_policy)

    def revenue(self, tax_policy, tax_amount):
        # aka tax
        for person in self.environment.people:
            if person.income > tax_policy:
                # Upper tax bracket
                person.income -= tax_amount[1]
                self.budget += tax_amount[1]
            else :
                # Lower tax bracket
                person.income -= tax_amount[0]
                self.budget += tax_amount[0]


        



        
        



pop_size = 2500
empty_ratio = 0.2
race_count = 3
religion_count = 5

sim1 = Govern(100000)

env1 = Environment(pop_size,empty_ratio, race_count, religion_count)
sim1.load_environment(env1)

races = [1]
religions = [0]

sentiment = []
budget = []
race_income = [env1.race_income]
religion_income = [env1.religion_income]


race_labels = ["White", "Black", "Asian"]
religion_labels = ["Christian", "Muslim", "Jewish", "Hindu", "Buddhist"]


plt.figure(figsize=(16,16))
fig, axs = plt.subplots(2,2)

axs[0,0].plot(np.arange(len(sentiment)), np.array(sentiment))

axs[0,1].plot(np.arange(len(budget)), np.array(budget))

axs[1,0].bar(race_labels, race_income[-1])

axs[1,1].bar(religion_labels, religion_income[-1])

pop = st.pyplot(plt)
progress_bar = st.progress(0)

n_iterations = st.sidebar.number_input("Number of Iterations", 3)

if st.sidebar.button('Run Simulation'):

    for i in range(n_iterations):
        success = sim1.create_policy(100, races,religions)
        sim1.revenue(500, [1,2])
        sim1.act()
        sentiment.append(sim1.get_revolutionary_sentiment())
        budget.append(sim1.budget)
        race_income.append(env1.race_income)
        religion_income.append(env1.religion_income)

        axs[0,0].plot(np.arange(len(sentiment)), np.array(sentiment))

        axs[0,1].plot(np.arange(len(budget)), np.array(budget))

        axs[1,0].bar(race_labels, race_income[-1])

        axs[1,1].bar(religion_labels, religion_income[-1])


        pop.pyplot(plt)
        plt.close("all")
        progress_bar.progress((i+1.)/n_iterations)

        print(race_income[-1])




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

        
