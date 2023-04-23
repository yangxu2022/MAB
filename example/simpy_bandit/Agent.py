import simpy
import numpy as np

from agent import Cable


class Agent(object):
    def __init__(self, env, adj_nodes, to_cables, from_cables, inner_cables, agent_id, agent_num, arm_real_means, arms):
        self.env = env
        self.adj_nodes = adj_nodes
        self.to_cables = to_cables
        self.from_cables = from_cables
        self.inner_cables = inner_cables
        self.agent_id = agent_id
        self.agent_num = agent_num
        self.arm_real_means = arm_real_means
        self.arms = arms
        self.decision_interval = 1
        self.com_interval = 1

        self.arm_num = len(self.arm_real_means)
        self.arm_means = np.zeros((self.agent_num, self.arm_num))
        self.arm_sns = np.zeros((self.agent_num, self.arm_num))
        self.arm_rws = np.zeros((self.agent_num, self.arm_num))

        self.arm_mean = np.zeros(self.arm_num)
        self.arm_sn = None
        self.arm_rw = None

        # UCB初始化需要pull每一个arm一次
        for arm_id in self.arms:
            rw = self._pull(arm_id)
            self._update_arm(arm_id, 1, rw)

        # 将事件加入env的进程中
        self.env.process(self.message_receiving())
        self.env.process(self.message_sending())
        self.env.process(self.decision_making())
        self.env.process(self.reward_arriving())

    def _update_arm(self, arm_id, sn, rw):

        self.arm_rws[self.agent_id][arm_id] += rw
        self.arm_sns[self.agent_id][arm_id] += sn
        self.arm_means[self.agent_id][arm_id] = self.arm_rws[self.agent_id][arm_id] / self.arm_sns[self.agent_id][
            arm_id]

    def _arm_selection(self):
        arm_sn = self.arm_sns[self.agent_id, self.arms]
        arm_mean = self.arm_means[self.agent_id, self.arms]
        c_i = np.sqrt(2 * np.log(self.env.now) / arm_sn)
        max_arm_index = np.argmax(arm_mean + c_i)
        return self.arms[max_arm_index]

    def _pull(self, arm_id):
        if np.random.random() < self.arm_real_means[arm_id]:
            return 1
        return 0

    def decision_making(self):
        while True:
            yield self.env.timeout(self.decision_interval)
            arm_id = self._arm_selection()
            rw = self._pull(arm_id)
            msg = self.msg_encoder(arm_id, 1, rw, self.agent_id)
            self.inner_cables[self.agent_id].put(msg)
            print(f'agent:{self.agent_id} pull arm:{arm_id} at:{self.env.now}')

    def reward_arriving(self):
        while True:
            msg = yield self.inner_cables[self.agent_id].get()
            arm_id, sample_number, reward, _ = self.msg_decoder(msg)
            if arm_id in self.arms:
                self._update_arm(arm_id, sample_number, reward)
                print(f'agent:{self.agent_id} get arm:{arm_id} reward at:{self.env.now}')

    def message_sending(self):
        """A process which generates messages."""
        while True:
            # wait for next transmission
            yield self.env.timeout(self.com_interval)
            self.com_interval = self.com_interval * 2
            for cable in self.to_cables:
                for arm_id in self.arms:
                    msg = {'arm_id': arm_id, 'from_agent': self.agent_id,
                           'sample_number': self.arm_sns[self.agent_id][arm_id],
                           'reward': self.arm_rws[self.agent_id][arm_id]}
                    cable.put(msg)
                    print(f'agent:{self.agent_id} send this at {self.env.now}, arm_id: {arm_id}')

    def message_receiving(self):
        """A process which consumes messages."""
        while True:
            # Get event for message pipe
            for cable in self.from_cables:
                msg = yield cable.get()
                arm_id = msg['arm_id']
                if arm_id not in self.arms:
                    continue
                sample_number = msg['sample_number']
                reward = msg['reward']
                from_agent = msg['from_agent']
                self.arm_sns[from_agent][arm_id] = sample_number
                self.arm_rws[from_agent][arm_id] = reward

    def get_arm_mean(self):
        self.arm_sn = np.sum(self.arm_sns, axis=0)
        self.arm_rw = np.sum(self.arm_rws, axis=0)
        self.arm_mean[self.arms] = self.arm_rw[self.arms] / self.arm_sn[self.arms]
        return self.arm_mean

    @staticmethod
    def msg_encoder(arm_id, sample_number, reward, from_agent):
        return {'arm_id': arm_id, 'sample_number': sample_number, 'reward': reward, 'from_agent': from_agent}

    @staticmethod
    def msg_decoder(msg):
        return msg['arm_id'], msg['sample_number'], msg['reward'], msg['from_agent']


if __name__ == '__main__':

    # 设定仿真持续时间
    SIM_DURATION = 500
    # Agent数量
    AGENT_NUMBER = 2
    # 初始化环境env
    env = simpy.Environment()

    # 每个Agent拥有的arm的集合
    arms_all = [[0, 1, 2], [0, 2, 3]]
    # 每个Arm的真实均值
    arm_real_means = np.array([0.5, 0.9, 0.8, 0.2])
    # 有向边+权值
    edges = [[0, 1, 1], [1, 0, 3]]

    # Agent的自身缓存
    inner_cable_delays = np.zeros(AGENT_NUMBER)
    inner_cables = np.array([Cable(env, delay) for delay in inner_cable_delays])

    # Agent的发送缓存
    to_cables_all = [[] for _ in range(AGENT_NUMBER)]
    # Agent的接收缓存
    from_cables_all = [[] for _ in range(AGENT_NUMBER)]

    # 邻接表
    adj_nodes_all = [[] for _ in range(AGENT_NUMBER)]

    # 根据有向边初始化邻接表、发送缓存和接收缓存
    for edge in edges:
        u, v, w = edge[0], edge[1], edge[2]
        adj_nodes_all[u].append(v)
        u_to_v_cable = Cable(env, w)
        to_cables_all[u].append(u_to_v_cable)
        from_cables_all[v].append(u_to_v_cable)

    # 初始化Agent
    agents = []
    for agent_id in range(AGENT_NUMBER):
        agents.append(
            Agent(env=env,
                  adj_nodes=adj_nodes_all[agent_id],
                  to_cables=to_cables_all[agent_id],
                  from_cables=from_cables_all[agent_id],
                  inner_cables=inner_cables,
                  agent_id=agent_id,
                  agent_num=AGENT_NUMBER,
                  arm_real_means=arm_real_means,
                  arms=arms_all[agent_id])
        )

    # 开始simulation
    env.run(until=SIM_DURATION)

    print("------")
    print(f'real mean:{arm_real_means}')
    print("------")
    print(f'agent:{agents[0].agent_id}, arm_mean:{agents[0].get_arm_mean()}')
    print("------")
    print(f'agent:{agents[1].agent_id}, arm_mean:{agents[1].get_arm_mean()}')
