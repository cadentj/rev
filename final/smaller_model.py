import random
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt

from collections import Counter

class Agent:

    def __init__(self, race, religion, income) :
        self.race = race     
        self.religion = religion
        self.income = income
        self.revolutionary_sentiment = 0        

    def update_revolutionary_sentiment(self, economy):
        self.revolutionary_sentiment = ((self.income * 0.5)/100000) + ((economy * 0.5)/1000)
        
    def __str__(self) -> str:
        return f'Race: {self.race}, Religion: {self.religion}, Income: {self.income}'
        
def norm(arr):
        return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def get_data(population_grid):
    x = []
    y = []
    race_data = []
    for i in range(len(population_grid)):
        for j in range(len(population_grid[i])):
            x.append(i)
            y.append(j)
            if population_grid[i][j] == -1:
                race_data.append(-1)
            else:
                race_data.append(population_grid[i][j].race)
    
    race_data = np.array(race_data)
    col = np.where(race_data == -1, 'w', np.where(race_data<1, 'b','r'))

    return x,y,col

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
                    else:
                        # Happy agents gain sentiment from their neighbors
                        neighborhood = neighborhood.flatten()
                        revolutionary_sentiment = np.average([neighbor.revolutionary_sentiment for neighbor in neighborhood if type(neighbor) == Agent])
                        person.revolutionary_sentiment = revolutionary_sentiment

     

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

    def step(self, budget):
        self.budget = budget
        self.scheilling()
        self.update_people()

    def update_policy(self, new_race_policy, new_religion_policy):
        self.race_policy = new_race_policy
        self.religion_policy = new_religion_policy

    def update_people(self):
        for person in self.people:
            # Enact policy
            benefit = np.average([self.race_policy[person.race], self.religion_policy[person.religion]])
            p = [1-benefit, benefit]

            choice = np.random.choice([-5,5], size=1, p=p)
            person.income += choice
            
            # Update race_income demographic
            race = person.race
            self.race_income[race] += choice

            # Update religion_income demographic
            religion = person.religion
            self.religion_income[religion] += choice

            # Update revolutionary sentiment
            person.update_revolutionary_sentiment(self.budget)



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

    def act(self):
        self.environment.step(self.budget)

    def get_revolutionary_sentiment(self):
        return np.average([person.revolutionary_sentiment for person in self.environment.people])

    def create_policy(self, cost, races, religions):
        self.budget -= cost
        if self.budget <= 0: return -1

        current_race_policy = self.environment.race_policy
        current_religion_policy = self.environment.religion_policy
        current_race_capital = self.environment.race_capital
        current_religion_capital = self.environment.religion_capital

        base_success = cost / self.budget
        print(base_success)
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

st.set_page_config(layout="wide")
st.title("Agent Based Model of Revolution")
st.sidebar.title("Parameters")

empty_ratio = st.sidebar.slider("Empty Ratio", 0.0, 0.5, 0.2)
race_count = st.sidebar.slider("Race Count", 1, 7, 3)
religion_count = st.sidebar.slider("Religion Count", 1, 7, 5)
n_iterations = st.sidebar.number_input("Number of Iterations", 3)

sim1 = Govern(100000)
env1 = Environment(pop_size,empty_ratio, race_count, religion_count)
sim1.load_environment(env1)

races = [1]
religions = [0]

sentiment = []
budget = []

race_income = [env1.race_income]
religion_income = [env1.religion_income]

race_policy = [env1.race_policy]
religion_policy = [env1.religion_policy]

race_capital = [env1.race_capital]
religion_capital = [env1.religion_capital]

race_labels = [i for i in range(race_count)]
religion_labels = [i for i in range(religion_count)]


initial_x, initial_y, initial_col = get_data(env1.population_grid)


# fig, axs = plt.subplots(4,2)
# fig.tight_layout(pad=5.0)
plt.style.use("ggplot")
plt.figure(figsize=(24, 8))

plt.subplot(131)
plt.axis('off')
plt.scatter(initial_x,initial_y,c=initial_col,marker='s',linewidth=0, s=100)
plt.title("Scheilling Model")

plt.subplot(263)
plt.plot(np.arange(len(sentiment)), np.array(sentiment), color='r')
plt.xlim(0,n_iterations)
plt.ylim(0,100)
plt.title("Stability")

plt.subplot(264)
plt.plot(np.arange(len(budget)), np.array(budget), color='r')
plt.xlim(0,n_iterations)
plt.ylim(0,1000000)
plt.title("Budget")

plt.subplot(265)
plt.bar(race_labels, race_income[-1], color='r')
plt.title("Income by Race")

plt.subplot(266)
plt.bar(religion_labels, religion_income[-1], color='r')
plt.title("Income by Religion")

plt.subplot(2,6,9)
plt.bar(race_labels, race_policy[-1], color='r')
plt.title("Policy by Race")

plt.subplot(2,6,10)
plt.bar(religion_labels, religion_policy[-1], color='r')
plt.title("Policy by Religion")

plt.subplot(2,6,11)
plt.bar(race_labels, race_capital[-1], color='r')
plt.title("Capital by Race")

plt.subplot(2,6,12)
plt.bar(religion_labels, religion_capital[-1], color='r')
plt.title("Capital by Religion")

pop = st.pyplot(plt)
progress_bar = st.progress(0)


if st.sidebar.button('Run Simulation'):

    for i in range(n_iterations):

        success = sim1.create_policy(100, races,religions)
        sim1.revenue(500, [1,2])

        sim1.act()

        sentiment.append(sim1.get_revolutionary_sentiment())
        budget.append(sim1.budget)
        race_income.append(env1.race_income)
        religion_income.append(env1.religion_income)

        race_policy.append(env1.race_policy)
        religion_policy.append(env1.religion_policy)

        race_capital.append(env1.race_capital)
        religion_capital.append(env1.religion_capital)

        new_x, new_y, new_col = get_data(env1.population_grid)

        plt.figure(figsize=(16,8))

        plt.subplot(121)
        plt.axis('off')
        plt.scatter(new_x,new_y,c=new_col,marker='s',linewidth=0, s=100)
        plt.title("Scheilling Model")

        plt.subplot(243)
        plt.plot(np.arange(len(sentiment)), np.array(sentiment), color='r')
        plt.xlim(0,n_iterations)
        plt.ylim(0,100)
        plt.title("Stability")


        plt.subplot(244)
        plt.plot(np.arange(len(budget)), np.array(budget), color='r')
        plt.xlim(0,n_iterations)
        plt.ylim(0,1000000)
        plt.title("Budget")

        plt.subplot(247)
        plt.bar(race_labels, race_income[-1], color='r')
        plt.title("Income by Race")

        plt.subplot(248)
        plt.bar(religion_labels, religion_income[-1], color='r')
        plt.title("Income by Religion")

        plt.subplot(247)
        plt.bar(race_labels, race_policy[-1], color='r')
        plt.title("Policy by Race")

        plt.subplot(248)
        plt.bar(religion_labels, religion_policy[-1], color='r')
        plt.title("Policy by Religion")

        plt.subplot(247)
        plt.bar(race_labels, race_capital[-1], color='r')
        plt.title("Capital by Race")

        plt.subplot(248)
        plt.bar(religion_labels, religion_capital[-1], color='r')
        plt.title("Capital by Religion")

        pop.pyplot(plt)
        plt.close("all")

        progress_bar.progress((i+1.)/n_iterations)

    st.balloons()



race_income