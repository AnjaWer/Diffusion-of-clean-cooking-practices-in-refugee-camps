# importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

################################################## METRICS ##########################################################

# defining metrics and other model-level data to track during simulation
def get_adopters(model):
    agent_stage = [agent.stage for agent in model.schedule.agents]
    adopters = sum(1 for i in agent_stage if i == 4) 
    return adopters

def get_stage1(model):
    agent_stage = [agent.stage for agent in model.schedule.agents]
    num = sum(1 for i in agent_stage if i == 1) 
    return num

def get_stage2(model):
    agent_stage = [agent.stage for agent in model.schedule.agents]
    num = sum(1 for i in agent_stage if i == 2) 
    return num

def get_stage3(model):
    agent_stage = [agent.stage for agent in model.schedule.agents]
    num = sum(1 for i in agent_stage if i == 3) 
    return num

def get_rejecters(model):
    agent_stage = [agent.stage for agent in model.schedule.agents]
    num = sum(1 for i in model.schedule.agents if (i.stage == 5))
    return num

def get_cf_cost(model):
    return model.cost_cf

def get_tf_cost(model):
    return model.cost_tf

#################################################### AGENT ###########################################################

class Household(Agent):
    """An agent representing 10 refugee households."""
    def __init__(self, unique_id, model, agent_group, strategy, initial_stage):
        super().__init__(unique_id, model)
        # initialize group (adopter category) according to DoI theory
        self.group = agent_group
        # inititalize decision strategy
        self.strategy = strategy
       
        # initialize social treshold and information threshold, depending on group (level of innovativeness)
        self.social_thresh = self.get_social_thresh(self.group)
        self.info_thresh = self.get_info_thresh(self.group)
        
        # initialize expected satisfaction, utility threshold and performance utility 
        self.satisfaction = np.nan 
        self.u_thresh = model.u_thresh        
        self.u_performance = np.nan
        
        # inititalize decision stage
        self.stage = initial_stage
        self.initial_adopter = 0
   
        # initialize information pool and awareness message
        self.info_pool = []
        self.message = 0

        # initialize income including cash transfer (for fuel) and ability-to-pay
        self.income = self.random.triangular(12000,24000,48000) + model.cash_transfer
        self.atp = model.atp
        
        self.time = 0
        
     ####################################### FUNCTIONS INCLUDED IN AGENT STEP #############################################
        
    # function returning the individual social threshold, based on adopter category 
    def get_social_thresh(self, group):
        # defining the social thresholds for each of the groups, adapted from Hidayatno et al. (2020)
        d = {"innovators": 0, 
            "early adopters": np.random.triangular(0, 0.03, 0.075), 
            "early majority": np.random.triangular(0.03, 0.075, 0.15),
            "late majority":  np.random.triangular(0.075, 0.15, 0.25),
            "laggards": np.random.triangular(0.1, 0.25, 0.4)}
        social_thresh = d[group] 
        return social_thresh
    
    # function returning the individual information theshold, based on adopter category 
    def get_info_thresh(self, group):
        d = {"innovators": 0.0, 
            "early adopters": 0.9, 
            "early majority": 0.95, 
             "late majority": 0.99, 
            "laggards": 1.0}
        info_thresh = d[group] 
        return info_thresh
   
    # function defining WoM communications from adopters to unaware agents 
    def send_message(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
        neighbors = [agent for agent in neighbors if (agent.stage == 1)]
        if len(neighbors) > 0:
            other_agent = self.random.choice(neighbors)
            other_agent.message = 1
                
    # function defining WoM communications from adopters/rejecters to social ties in stages 2 and 3
    def update_info(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        neighbors = self.model.grid.get_cell_list_contents(neighbors_nodes)
        neighbors = [agent for agent in neighbors if (agent.stage == 4) or (agent.stage == 5)]
        temp_pool = [self.satisfaction]
        
        if len(neighbors) > 0:
            for agent in neighbors:
                self.info_pool.append(agent.satisfaction)
        if np.isnan(self.satisfaction) == False:
            for info in self.info_pool:
                # count double if the new information is below the agent's current expected satisfaction
                if info < self.satisfaction:
                    temp_pool.append(info)
                    temp_pool.append(info)
                else:
                    temp_pool.append(info)
        if len(self.info_pool) > 0:
            self.satisfaction = np.mean(self.info_pool)
        self.info_pool = []

    
    # function returning True if social threshold is met, else False
    def check_social_thresh(self):
        adopters = [agent for agent in self.model.schedule.agents if (agent.stage == 4) and (agent.group != 'early adopters')]
        # early adopter agents count 1.5 times
        early_adopters = [agent for agent in self.model.schedule.agents if (agent.stage == 4) and (agent.group == 'early adopters')]  
        social_value = len(adopters) + 1.5*len(early_adopters)
        if social_value >= self.social_thresh*self.model.num_agents:
            return True
        else:
            return False    
    
    # function returning True if expected satisfaction meets information threshold, else False
    def check_expected_satisfaction(self):
        if self.satisfaction >= self.info_thresh:
            return True
        else:
            return False    
         
    # function returning 1 if clean fuel cost are below (time-discounted) traditional fuel cost, else returning 0 - for cost-optimizing strategy
    def choose_cheapest(self):
        cost_diff = self.model.cost_cf - self.model.cost_tf
        if cost_diff < 0:
            self.economic_barrier = 1
            return 1
        else: 
            self.economic_barrier = 0
            return 0
        
    # function returning 1 if economic barrier is overcome, else returning False
    def check_cost(self):
        cost_diff = self.model.cost_cf - self.model.cost_tf
        if (cost_diff <= self.atp * self.income):
            self.economic_barrier = 1
            return 1
        else:
            self.economic_barrier = 0
            return 0    
    
    # function returning 1 if clean fuel is available, based on probability, else returning 0
    def check_availability(self):
        if self.random.random() < self.model.p_supply_delay:
            return 0
        else:
            return 1
        
    # function returning 1 if positive performance, else 0, based on probability
    def performance_event(self):
        if self.random.random() < self.model.p_bad_performance:
            return 0
        else: 
            return 1
            
    # function for adopter agents, calculating the current satisfaction value based on cost, availability, performance
    def update_satisfaction(self):
        u_availability = self.check_availability()
        u_cost = self.check_cost()
        self.u_performance = self.performance_event()
        self.satisfaction = sum([(1/3)*u_cost, (1/3)*self.u_performance, (1/3)*u_availability])
    
    # function returning True, if satisfaction value above time-discounted utility threshold
    def check_satisfaction(self):
        if self.satisfaction >= self.u_thresh:
            return True
        else:
            return False

    ################################################# AGENT STEP ########################################################### 
    
    # agents perform actions based their current stage 1 to 5, and strategy
    def step(self):
        self.time = self.model.schedule.time
        
        if self.stage == 1:
            if self.message == 1:
                self.stage = 2             
        elif self.stage == 2:
            if self.check_social_thresh():
                if self.check_availability():
                    if self.strategy == "imitators":
                        self.stage = 4
                    else:
                        self.stage = 3
        elif self.stage == 3:
            if self.random.random() < self.model.f_decide:
                self.update_info()
                cost = self.check_cost()
                if self.check_availability():
                    if self.strategy == "optimizers":
                        if self.choose_cheapest():
                            self.stage = 4
                    elif self.strategy == "advice seekers":
                        if self.check_expected_satisfaction():
                            self.stage = 4
                    else:
                        if self.check_expected_satisfaction() and (cost == 1):
                            self.stage = 4
        elif self.stage == 4:
            self.send_message()
            self.update_satisfaction()
            # if initial adopter, remain adopter, as long as fuel is available
            if (self.initial_adopter == 1) and self.check_availability():
                pass
            else:
                if self.check_satisfaction() == False:
                    self.stage = 5
        elif self.stage == 5:
            if self.random.random() < self.model.f_decide:
                if self.check_availability() and (self.u_performance == 1) and self.check_cost():
                    self.stage = 4
        else:
            return "invalid agent stage"
        
################################################# MODEL ########################################################### 
    
class DiffusionModel(Model):
    """This model represents a refugee camp with households as agents. 
    The agents go through decision stages, deciding whether or not to adopt and keep using clean cooking fuels."""
    def __init__(self, seed, initial_adopters, cash_transfer, vouchers, info_campaign, maintenance_capacity, 
                 N, avg_node_degree, p_rewiring, r_discount, atp, p_bad_performance, p_imitators, p_optimizers, p_advice_seekers, 
                 p_supply_delay, price_tf, price_cf, price_shock, n_cf_price_shock):
        self._seed = seed
        self.random.seed(seed)
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.running = True
        
        # generate social network
        self.avg_node_degree = avg_node_degree
        self.p_rewiring = p_rewiring
        self.G = nx.connected_watts_strogatz_graph(n = self.num_agents, k = avg_node_degree, p = p_rewiring, seed=42)
        self.grid = NetworkGrid(self.G)
        
        # initialize cost, considering time-discounting and price subsidies
        self.r_discount = r_discount
        self.initial_cost_tf = sum([price_tf, (1/(1+r_discount)**1)*price_tf, (1/(1+r_discount)**2)*price_tf, (1/(1+r_discount)**3)*price_tf])
        self.vouchers = vouchers
        self.initial_cost_cf = price_cf
        self.cost_cf = self.initial_cost_cf - self.vouchers
        self.cost_tf = self.initial_cost_tf
        self.price_shock = price_shock
        self.n_cf_price_shock = n_cf_price_shock
    
        # initialize levers
        self.cash_transfer = cash_transfer
        self.n_info_campaign = info_campaign
        self.maintenance_capacity = maintenance_capacity
      
        # initialize further model parameters
        self.initial_adopters = initial_adopters
        self.f_decide = 0.25
        self.u_thresh = 1
        self.atp = atp
        self.p_bad_performance = p_bad_performance
        self.p_supply_delay = p_supply_delay
        self.t_shock = 0
        
        # initialize ratios per decision strategy
        self.p_imitators = p_imitators
        self.p_optimizers = p_optimizers
        self.p_advice_seekers = p_advice_seekers
        
        # collecting model-level and agent-level data
        self.datacollector = DataCollector(
            model_reporters={"Adoption": get_adopters, "Rejection": get_rejecters, 
                             "Decision": get_stage3, "Awareness": get_stage2, "Ignorance": get_stage1, 
                             "CF cost": get_cf_cost, "TF cost": get_tf_cost}, 
            agent_reporters={"Group": lambda a: a.group, "Strategy": lambda a: a.strategy, "Stage": lambda a: a.stage, 
                             "Satisfaction": lambda a: a.satisfaction, "Performance": lambda a: a.u_performance}) 
        
        # define heterogenous groups, and create their agents
        groups = {1: "innovators", 2: "early adopters", 3: "early majority", 4: "late majority", 5: "laggards"}    
        strategies = {1: "imitators", 2: "optimizers", 3: "advice seekers", 4: "deliberators"}   
        for i, node in enumerate(self.G.nodes()):    
            g = groups[int(np.random.choice([1, 2, 3, 4, 5], 1, p=[0.025, 0.135, 0.34, 0.34, 0.16]))]
            s = strategies[int(np.random.choice([1, 2, 3, 4], 1, p=[p_imitators, p_optimizers, p_advice_seekers, 
                                                                    max(0,1-p_imitators-p_optimizers-p_advice_seekers)]))]
            a = Household(i, self, g, s, 1)
            self.schedule.add(a)
            self.grid.place_agent(a,node)     
        
        # define some initial adopters
        adopter_agents = self.random.sample(self.schedule.agents, int(self.initial_adopters*self.num_agents))
        for a in adopter_agents:
            a.stage = 4
            a.u_performance = 1
            a.initial_adopter = 1
            
########################################### FUNCTIONS INCLUDED IN MODEL STEP ################################################
    
    def update_cost_cf(self):
        new_cost = ((1+self.price_shock)*self.initial_cost_cf - self.vouchers)
        return new_cost
          
    def maintenance(self):
        rejecters = [agent for agent in self.schedule.agents if (agent.stage == 5)] 
        rejecter_sample = self.random.sample(rejecters, min(len(rejecters), self.maintenance_capacity))
        for agent in rejecter_sample:
            agent.u_performance = 1
    
    def info_campaign(self):
        nonadopters = [agent for agent in self.schedule.agents if (agent.stage == 3) or (agent.stage == 2) or (agent.stage == 1)]
        nonadopter_sample = self.random.sample(nonadopters, min(len(nonadopters), self.n_info_campaign))
        for agent in nonadopter_sample:
            agent.message = 1
            agent.info_pool.append(1)
    
################################################# MODEL STEP ########################################################### 
    
    def step(self):
        # run info campaign intervention, in the first 50 time steps
        if self.schedule.time < 50:
            self.info_campaign()
        
        # maintenance intervention
        self.maintenance()

        # agent steps
        self.schedule.step()  
        
        # collect data
        self.datacollector.collect(self)
        
        # update cost 
        if self.n_cf_price_shock > 0:
            if (self.schedule.time % int(150/(1+self.n_cf_price_shock)) == 0):
                self.t_shock = self.schedule.time
                self.cost_cf = self.update_cost_cf() 
            if (self.schedule.time - self.t_shock == 12):
                self.cost_cf = self.initial_cost_cf - self.vouchers
          
        # stop the model after 150 time steps (for the user interface)
        # not necessary for experimentation
        if self.schedule.time >= 150:
            self.running = False
        
    def run_model(self, n):
        for i in range(n):
            self.step()
            
            
    