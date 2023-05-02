import networkx as nx
import numpy as np
import collections

# class Graph:

class Agent:

    def __init__(self, location, race, age, religion, gender, type) :
        self.race = race       # Random distribution of 5
        self.age = age      # STD
        self.religion = religion    # Dependent
        # is gender an important factor? 
        self.gender = gender    # M/F
        self.type = type    # Ruler, elite, or person


class PopCenterNode:
    def __init__ (self, id, population_count) :
        self.id = id
        self.population_count = population_count
        # empty on init to prep for add_person method
        self.population = []



    def initial_demographic_count(self):
        race_array = []
        age_array = []
        religion_array = []
        
        for person in self.population:
            race_array.append(person.race)
            age_array.append(person.age)
            religion_array.append(person.religion)

        race = collections.Counter(race_array)
        age = collections.Counter(age_array)
        religion = collections.Counter(religion_array)

    def add_person (self, agent):
        self.population.append(agent)

    def __str__(self):
        return str(self.id)
    
    

class Environment:
    def __init__(self, population_density, area, population_centers) :
        # How dense the cities/towns are, scale of 1-10
        self.population_density = population_density
        self.population_center_count = population_centers

        # issue of making many small populations?
        population_centers_distr = np.random.normal(population_density/10, 0.3, self.population_center_count)
        # Density * area = population
        self.population_centers = [int(abs(i) * area) for i in population_centers_distr]


        self.population_centers_graph = nx.Graph()
        self.population_dict = {}


        for node_count, population in enumerate(self.population_centers):
            self.population_dict[node_count] = PopCenterNode(node_count, self.population_centers[node_count])

        
            self.population_centers_graph.add_node(node_count)


            for i in range(population):
                # need to refine location to grid
                # placeholders
                person = self.create_agent(node_count, 5, 5)

                self.population_dict[node_count].add_person(person)

            self.population_dict[node_count].initial_demographic_count()
            


            # set edges and connections

        


        

    def create_agent(self, location, race_count, religion_count):
        race = np.random.randint(race_count)
        # look up normal popualtion age distribution
        age = int(abs(np.random.normal(0.5, 0.3)) * 100)
        religion = np.random.randint(religion_count)
        gender = np.random.randint(2)
        type = "placeholder"

        return Agent(location, race, age, religion, gender, type)



    def organize(self) :
        return self.population_centers_count, self.population_center_count
    
    
    def demographics(self):
        return 0

    # def update_population():


    #     return 0

        # update the popluation every cycle based on spatial factors
        # people tend to associate with others of similar attributes
        # larger groups of attributes carry more power in the government
        

        # ASSUMPTION: the demographics of a population are influenced by the relative strength of certain attributes in government
        # ASSUMPTION: It is harder to make a living and therefore successfully reproduce under policy that does not benefit one's attributes
    
    

class Govern:

    def __init__(self, size, receptiveness, environment):
        
        self.size = size
        
        self.receptiveness = receptiveness

        # Government receptiveness
        # government makeup
        # government size - monarchy (1), 
        # governments size swings towards smaller or larger.
        # smaller try to stay small
        # big try to stay big
        # medium size are in a careful equilibrium
        
        # assumption: governments pass policy to benefit the make up of the government
        # balanced with the makeup of the local governments
        # balanced with the makeup of the elites
        # and finally the people
        self.environment = environment 

        

    def pass_policy():
        # only looking at policy on the national level because local level policies don't really influence at the end of the day
        # pass policy to increase income or ease of living for one attribute
        # policy success rate determined by the effectiveness of the government? 
        # policy success rate is null - not really important in a geneational model - might as wlel just implement success rate as an influence of those affected

        # ASSUMPTION: the effectiveness of a government is determined by a variety factors which will be mainly encapsulated by one variable
        # Examples: reflection in poeple, strengths of organizations

        

        return ["group"]

        


    # def update_government():
    #     # every couple of cycles
    #     # government update is random
    #     # tends to move to reflect people in more receptive governments
    #     # tends to move to reflect itself in less reflective governmetns


        


# population = 10000
# land_space = 1000000
# random_seed = 1

# # Normal log distribution
# gov_size = 1
# # Random
# gov_receptiveness = 5


# generations = 100


# for i in range(generations ) :
#     print("a")


