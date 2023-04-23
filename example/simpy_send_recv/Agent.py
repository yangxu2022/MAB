import simpy

from agent import Cable


class Agent(object):
    def __init__(self, env, to_cables, from_cables, agent_id):
        self.env = env
        self.to_cables = to_cables
        self.from_cables = from_cables
        self.agent_id = agent_id
        self.number = 1
        self.send_flag = 1
        self.env.process(self.receive())
        self.env.process(self.send())


    def send(self):
        """A process which randomly generates messages."""
        while True:
            # wait for next transmission
            yield self.env.timeout(1)
            if self.send_flag == 0:
                for cable in self.to_cables:
                    msg = {'num': self.number + 1, 'from_agent': self.agent_id}
                    cable.put(msg)
                    self.send_flag = 1
                # print(f'agent:{self.agent_id} send this at {self.env.now}, num: {self.number + 1}')

    def receive(self):
        """A process which consumes messages."""
        while True:
            # Get event for message pipe
            for cable in self.from_cables:
                msg = yield cable.get()
                self.number = msg['num']
                self.send_flag = 0
                print(f'agent:{self.agent_id} received this at {self.env.now}, num: {self.number}')


if __name__ == '__main__':
    SIM_DURATION = 100
    env = simpy.Environment()

    agent_1_2_cable = Cable(env, 1)
    agent_2_1_cable = Cable(env, 5)
    agent_1 = Agent(env, [agent_1_2_cable], [agent_2_1_cable], '1')
    agent_2 = Agent(env, [agent_2_1_cable], [agent_1_2_cable], '2')
    agent_1.send_flag = 0

    env.run(until=SIM_DURATION)