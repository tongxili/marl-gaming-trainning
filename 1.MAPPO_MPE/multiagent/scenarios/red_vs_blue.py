import numpy as np
from multiagent.core import World, Agent, Landmark, Rule_based_Agent
from multiagent.scenario import BaseScenario
from multiagent.utils import four_dir_generate_random_coordinates

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        #配置无人机个数，环境维度
        world.dim_c = 2
        
        num_agents = 4 # red
        num_rule_agents = 5 # blue
        num_landmarks = 0 # TEMP: no obstacles
        num_food = 1
        self.num_rule_agents = num_rule_agents
        self.num_good_agents = num_agents
        self.num_food = num_food
        #对无人机进行状态设置等等。。。
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.accel = 0.8
            agent.size = 0.02
            agent.max_speed = 0.5
            agent.death = False
            agent.pusai = np.pi
            agent.hit = False
            agent.movable = True
        
        # rule_based_agents
        world.rule_agents = [Rule_based_Agent() for _ in range(num_rule_agents)]
        for i, rule_agent in enumerate(world.rule_agents):
            rule_agent.name = 'agent %d' % i
            rule_agent.collide = True
            rule_agent.silent = True
            rule_agent.accel = 0.8
            rule_agent.size = 0.02
            rule_agent.max_speed = 0.5 # 实际发现太快了，改小一点
            rule_agent.death = False
            rule_agent.pusai = np.pi
            rule_agent.hit = False
            rule_agent.movable = True
            rule_agent.action.u = np.zeros(world.dim_p) # initialize action.u
            rule_agent.action.c = np.zeros(world.dim_c)

        #从上到下为food 0 1 2
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False
            landmark.death = False
        
        #障碍物
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.025
            landmark.boundary = False
            landmark.death = False
        
        world.landmarks += world.food
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        #一轮就一次
        for agent in world.agents:
            agent.color = np.array([255, 0, 0])
            #只是设置一下颜色！！
            #之后的刷新场面最好不要随机！
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0, 255, 255])

        for i, landmark in enumerate(world.food):
            landmark.color = np.array([1, 1, 0])
        
        # generate random positions
        screen_size = [-0.9, 0.9]
        blue_pos, red_pos, food_pos = four_dir_generate_random_coordinates(screen_size=screen_size, num_rule=self.num_rule_agents, dist_rule=0.06,
                                                                           num_adv=self.num_good_agents, num_food=self.num_food)
        # print("Pos: ", blue_pos[0], food_pos[0])
        for rule_agent in world.rule_agents:
            rule_agent.color = np.array([0, 0, 255])
            
        for i, agent in enumerate(self.good_agents(world)):
            # interval = 2.0 / (len(self.good_agents(world)) + 1)
            #print(interval)，中心距离为0.5
            # agent.state.p_pos = np.array([-0.9, 1 - (i+1)*interval])
            agent.state.p_pos = red_pos[i]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
            agent.death = False
            agent.hit = False
        
        for i, rule_agent in enumerate(self.rule_agents(world)):
            # interval = 2.0 / (len(self.rule_agents(world)) + 1)
            # rule_agent.state.p_pos = np.array([0.8, 1 - (i+1)*interval])
            rule_agent.movable = True
            rule_agent.state.p_pos = blue_pos[i]
            rule_agent.state.p_vel = np.zeros(world.dim_p)
            rule_agent.state.c = np.zeros(world.dim_c)
            
            rule_agent.death = False
            rule_agent.hit = False

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary: # other landmarks
                # Random position and 0 velocity for landmarks
                landmark.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p) # 速度为0
                # if landmark.name == 'landmark 0':
                #     landmark.state.p_pos = np.array([-0.1,-0.8])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 1':
                #     landmark.state.p_pos = np.array([0.1,-0.7])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 2':
                #     landmark.state.p_pos = np.array([0.1,0.3])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 3':
                #     landmark.state.p_pos = np.array([-0.1,0.1])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 4':
                #     landmark.state.p_pos = np.array([-0.2,0.4])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 5':
                #     landmark.state.p_pos = np.array([0.2,0.8])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 6':
                #     landmark.state.p_pos = np.array([0.3,0.5])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 7':
                #     landmark.state.p_pos = np.array([0.6,0.7])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 8':
                #     landmark.state.p_pos = np.array([0.6,0.1])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 9':
                #     landmark.state.p_pos = np.array([0.7,-0.2])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 10':
                #     landmark.state.p_pos = np.array([0.8,-0.5])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 11':
                #     landmark.state.p_pos = np.array([0.9,-0.8])+np.random.uniform(-0.1, +0.1, world.dim_p)
                # elif landmark.name == 'landmark 12':
                #     landmark.state.p_pos = np.array([0.9,-0.8])+np.random.uniform(-0.1, +0.1, world.dim_p)

        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = food_pos[i] # change for 4 quardants
            landmark.state.p_vel = np.zeros(world.dim_p)
        
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist <= dist_min else False

    def good_agents(self, world):
        return [agent for agent in world.agents]
    
    def rule_agents(self, world):
        return [agent for agent in world.rule_agents]

    def reward(self, agent, world): # reward for red object
        main_reward = self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # ADD: agent reward for red agents, adding capture of blue
        if agent.death:
            return 0
        
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            rew = -5 # 越界扣分
        else:
            rew = 0 #
        
        # for target in world.food:
        #     # 距离越远，reward越小
        #     rew -= 10 * np.sqrt(np.sum(np.square(agent.state.p_pos - target.state.p_pos)))
            
        
        for obstacle in world.landmarks:
            # 和landmarks非常近时，奖励相应增加或减少
            # 暂时没用到障碍物landmark
            # if np.sqrt(np.sum(np.square(obstacle.state.p_pos - agent.state.p_pos)))<0.1:
            #     if 'food' in obstacle.name:
            #         rew -= 5 # 不能太近
            #     elif 'landmark' in obstacle.name:
            #         rew -= 5

            if self.is_collision(obstacle,agent):
                if 'landmark' in obstacle.name:
                    rew-=3
        
        for other_agent in world.agents:
            if agent == other_agent: continue
            if np.sqrt(np.sum(np.square(other_agent.state.p_pos - agent.state.p_pos)))<0.1:
                rew -= 1 # 红方相互之间不能太近
            if self.is_collision(agent, other_agent):
                rew -= 2 # 红方不能相撞
        
        num_rule_agents = len(world.rule_agents)
        for i, rule_agent in enumerate(world.rule_agents):
            if rule_agent.death: continue
            # 红方和蓝方，距离越近，reward越大
            dist_red = np.sqrt(np.sum(np.square(agent.state.p_pos - rule_agent.state.p_pos)))
            if dist_red<0.1:
                rew += 1
            rew += 2.5 * np.exp(-dist_red) / num_rule_agents # 平均一下

            if self.is_collision(agent, rule_agent):
                rew += 10
                if i==0:
                    rew += 10 # 碰撞leader，奖励更多
                agent.death = True
                # agent.movable = False
                rule_agent.death = True
                rule_agent.color = np.array([0, 0, 0])
                rule_agent.movable = False
            
            # 蓝方和目标点
            for target in world.food: 
                dist_food = np.sqrt(np.sum(np.square(target.state.p_pos - rule_agent.state.p_pos)))
                # rew += 2 * dist_food # 蓝方距离目标点越远，reward越大
                rew -= 5 * np.exp(-dist_food) / num_rule_agents # 蓝方距离目标点越近，reward扣越多
                
                # 蓝方到达目标点，扣分
                if self.is_collision(rule_agent, target):
                    # print("blue wins!")
                    rew -= 10
                    rule_agent.death = True
                    rule_agent.movable = False
                    self.rule_agents(world)[0].death = True # leader也到达
                    self.rule_agents(world)[0].movable = False
        
        #这一段训练后期再加
        #for i, landmark in enumerate(world.food):
        #    if(landmark.color == ([1, 0, 0])):
        #                finished = True
        #                print("赢了！")
                        #rew+=1500
                        #return rew
            #self.reset_world(world)


        def bound(x):
            if x < 0.9:
                return 0
            if (x < 1.0 and x > 0.9 ):
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 3 * bound(x)

        return rew

    def observation(self, agent, world):
        #这个是用来观测的
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        comm = []
        rule_pos = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)
        
        for rule_agent in world.rule_agents:
            rule_pos.append(rule_agent.state.p_pos - agent.state.p_pos)
            # 暂时不把rule_agent的速度加进来
        
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + rule_pos + entity_pos + other_pos + other_vel)

    def finish(self, world):
        agents = self.good_agents(world)
        for i, landmark in enumerate(world.food):
            for good_a in agents:
                if self.is_collision(good_a, landmark):
                    return True
        return False