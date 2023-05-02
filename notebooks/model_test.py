import random
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Agent:

    def __init__(self, location, race, age, religion) :
        self.race = race       # Random distribution of 5
        self.religion = religion    # Dependent
        self.type = type    # Ruler, elite, or person

    def get_attribute(self, attribute):
        if (attribute == "race"):
            return self.race
        elif (attribute == "religion"):
            return self.religion
        elif (attribute == "type"):
            return self.type
        else:
            return 0


    

class Environment:

    def __init__(self, size, empty_ratio, race_count, religion_count):
        self.size = size 
        population_size = int(np.sqrt(self.size))**2
        self.dim = int(np.sqrt(population_size))

        population = self.generate_population(population_size, empty_ratio, race_count, religion_count)
        self.population = np.reshape(population, (int(np.sqrt(population_size)), int(np.sqrt(population_size))))



    def generate_population(self, population_size, empty_ratio, race_count, religion_count):
        p = [1-empty_ratio, empty_ratio]
        choice = np.random.choice([0,1], size=population_size, p=p)
        population = [self.create_agent(race_count, religion_count) if i == 0 else -1 for i in choice]
        return population

    def create_agent(self, race_count, religion_count):
        race = np.random.randint(race_count)
        # look up normal popualtion age distribution
        age = int(abs(np.random.normal(0.5, 0.3)) * 100)
        religion = np.random.randint(religion_count)
        gender = np.random.randint(2)
        type = "placeholder"

        return Agent(race, age, religion, gender)
    
    def get_grid(self, attribute) :

        return [[person.get_attribute(attribute) if type(person) == Agent else -1 for person in row] for row in self.population]


st.title("Schelling's Model of Segregation")

race_count = st.sidebar.slider('How many races to include?', 1, 5, 2)
religion_count = st.sidebar.slider('How many religions to include?', 1, 7, 5)
empty_ratio = st.sidebar.slider('How many empty plots?', .0, 0.5, 0.1)
size = st.sidebar.slider('Size?', 100,10000, 400)


option = st.sidebar.selectbox(
    'Agent Display',
    ('race', 'religion', 'Gender'))


env = Environment(size, empty_ratio, race_count, religion_count)

#Plot the graphs at initial stage
plt.style.use("ggplot")
plt.figure(figsize=(10, 4))

x = 


# Left hand side graph with Schelling simulation plot
colors = ['red', 'royalblue', 'green', 'orange', 'black', 'purple']
st.sidebar.write(colors[0:race_count] + ['white'])
cmap = ListedColormap(colors[0:race_count] + ['white'])
plt.subplot(121)
plt.axis('off')
plt.pcolor(env.get_grid(option), cmap=cmap, edgecolors='w', linewidths=1)
plt.colorbar()

city_plot = st.pyplot(plt)

if st.sidebar.button('Run Simulation'):
    city_plot.pyplot(plt)

