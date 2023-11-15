import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np
import random
from simulator import Step_HRRO
# from ray.rllib.env.env_context import EnvContext


class HRROenv(gym.Env):
    """
        The environment used to train an agent to control the batch/semibatch RO
        process considering BIA's pilot goals and constraints
    """

    def __init__(self, env_config):

        # Only concentrations by now
        # self.conductivity: float =
        # self.conductivity_permeate: float =
        # self.conductivity_brine: float =

        # self.feed_water_volume: float = 2500

        # concentration NaCl having same osmotic pressure (60 bar) as electrolyte bath
        self.electrolyte_concentration: float = 76
        # self.concentration = self.electrolyte_concentration / 15
        # self.concentration_permeate: float = 0
        # self.concentration_brine: float = 0

        # self.temperature: float = 298.15

        # self.pressure_recirc: float = 0
        # self.pressure_supply: float = 0
        # self.pressure_batch

        # self.flowrate_feed: float = 9 / 60
        # self.flowrate_recirc: float = 54 / 60
        self.permeate_flowrate_min: float = 10  # [L/min]

        # Generate process instance with BIA's parameters
        membrane = env_config['membrane']
        solution = env_config['solution']
        design = env_config['design']
        operation = env_config['operation']
        self.process = Step_HRRO(membrane, solution, design, operation)
        self.process.initialize_volume_energy_semibatchtime()
        self.workexchanger_initial_volume = self.process.work_exchanger_initial_volume
        self.process_initial = self.process
        self.process_initial.initialize_volume_energy_semibatchtime()

        # Observation space
        # self.min_conductivity: float =
        self.min_feed_water_volume: float = 200
        self.min_concentration: float = self.electrolyte_concentration / 20
        self.min_temperature: float = 28 + 273.15
        self.min_pressure: float = 0

        low_obs = np.array([-1, -1, -1, -1, -1])

        # self.max_conductivity: float =
        self.max_feed_water_volume: float = 2500
        self.max_concentration: float = self.electrolyte_concentration / 10
        self.max_temperature: float = 32 + 273.15
        self.max_pressure: float = 125

        high_obs = np.array([1, 1, 1, 1, 1])

        self.observation_space = Box(
                low=low_obs,
                high=high_obs,
                shape=(5,),
                dtype=np.float32
        )

        # Action space
        self.min_flowrate_feed: float = 4
        self.min_flowrate_recirc: float = 20
        self.min_purge_time: float = 0.5

        low_actions = np.array(
            [
                self.min_flowrate_feed / 60,
                self.min_flowrate_recirc / 60,
                self.min_purge_time
            ]
        )

        self.max_flowrate_feed: float = 15
        self.max_flowrate_recirc: float = 70
        self.max_purge_time: float = 1.5

        high_actions = np.array(
            [
                self.max_flowrate_feed / 60,
                self.max_flowrate_recirc / 60,
                self.max_purge_time
            ]
        )

        self.action_space = Dict({
            'flowrates_pgtime': Box(
                low=low_actions,
                high=high_actions,
                shape=(3,),
                dtype=np.float32
            ),
            'end_semibatch': Discrete(2),
            'end_process': Discrete(2)
        })

        # Normalization variables

        self.mean_volume = (
            self.max_feed_water_volume + self.min_feed_water_volume
        ) / 2
        self.mean_concentration = (
            self.max_concentration + self.min_concentration
        ) / 2
        self.mean_temperature = (
            self.max_temperature + self.min_temperature
        ) / 2
        self.mean_pressure = (
            self.max_pressure + self.min_pressure
        ) / 2

    def step(self, action):  # sourcery skip: simplify-numeric-comparison

        reward = 0

        if self.process.phase == 'purge&refill':
            return self.end_cycle_step(action)

        if self.process.phase == 'semibatch':
            if action['end_semibatch']:
                self.switch_to_batch()
            elif self.process.semibatch_current_time >= 3600:
                reward -= 5 / 1000
                self.switch_to_batch()

        self.process.set_agent_decisions(
            flow_supply=action['flowrates_pgtime'][0],
            flow_recirc=action['flowrates_pgtime'][1]
        )

        self.process.step()

        if self.process.check_alpha() < 1.5:
            reward -= 1 / 1000

        end = False
        if self.process.feed_tank_current_volume <= 250:
            reward -= 1 / 2
            end = True

        if self.process.pressure_supply / 100 >= 120:
            reward = -1
            end = True
        elif self.process.pressure_supply > 11500:
            reward -= (self.process.pressure_supply / 100 - 114)**3 / 1000

        observation = self._get_obs_norm()

        return observation, reward, end, False, {}

    def end_cycle_step(self, action):

        reward = 0

        if self.process.calculate_permeate_flow_rate() < self.permeate_flowrate_min:
            reward -= 0.01

        concentrations = self.process.concentrations
        if concentrations.mean(axis=0)[0] < self.electrolyte_concentration:
            reward -= 0.02    # the process aims to meet the initial concentration
        elif concentrations.mean(axis=0)[0] >= self.electrolyte_concentration * 1.2:
            reward -= 0.02    # avoid exceeding the target concentration by more than 20%

        self.process.flow_supply_purge = self.process.flow_supply_batch
        self.process.flow_supply_purge_previous = self.process.flow_supply_batch
        self.process.set_agent_decisions(
            flow_supply=action['flowrates_pgtime'][0],
            purge_time=action['flowrates_pgtime'][2]
        )

        self.process.step()

        end = False
        if self.process.feed_tank_current_volume <= 250:
            reward -= 0.1
            end = True

        self.process.calculate_SEC()
        if self.process.SEC_Total < 2.5 and self.process.SEC_Total > 0:  # kWh per cycle
            reward += (4 - self.process.SEC_Total)**2 / 1000
        elif self.process.SEC_Total >= 2.5:
            reward -= 5 / 1000
        else:
            reward -= 0.1

        if (
            not end
            and self.process.feed_tank_current_volume < 700
            and action['end_process'] == 1
        ):
            end = True
            if self.process.feed_tank_current_volume > 300:
                reward -= (self.process.feed_tank_current_volume - 250) / 1000
            else:
                reward += 0.01

        observation = self._get_obs_norm()

        self.process.initialize_volume_energy_semibatchtime()

        return observation, reward, end, False, {}

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.process = self.process_initial
        self.process.initialize_volume_energy_semibatchtime()

        self.process.feed_tank_current_volume = random.randint(1500, 2500)
        self.process.concentration_feed = (
            np.array([self.electrolyte_concentration / random.randint(10, 20), 0])
        )
        self.process.reset_concentrations()
        self.process.temperature = (28 + random.random() * 4) + 273.15
        self.process.pressure_supply = 0
        self.process.pressure_recirc = 0
        self.process.flow_supply_semibatch = (
            random.randint(self.min_flowrate_feed, self.max_flowrate_feed)
        ) / 60
        self.process.flow_supply_semibatch_previous = self.process.flow_supply_semibatch
        alpha = 1.5 + random.random() * 4.5
        self.process.flow_recirc_semibatch = (
            np.clip(
                self.process.flow_supply_semibatch * alpha,
                self.min_flowrate_recirc / 60,
                self.max_flowrate_recirc / 60
            )
        )
        self.process.flow_recirc_semibatch_previous = self.process.flow_recirc_semibatch

        self.process.current_cycle = 1
        self.process.time = 0
        self.process.time_start = 0
        self.process.time_agent_last = 0

        self.process.phase = 'semibatch'

        observation = self._get_obs_norm()

        return observation, {}

    def _get_obs_norm(self):
        volume_norm = (
            self.process.feed_tank_current_volume
            - self.mean_volume
        ) / (self.mean_volume - self.min_feed_water_volume)
        concentration_norm = (
            self.process.concentration_feed[0]
            - self.mean_concentration
        ) / (self.mean_concentration - self.min_concentration)
        temperature_norm = (
            self.process.temperature
            - self.mean_temperature
        ) / (self.mean_temperature - self.min_temperature)
        pressure_supply_norm = (
            self.process.pressure_supply / 100
            - self.mean_pressure
        ) / self.mean_pressure
        pressure_recirc_norm = (
            self.process.pressure_recirc / 100
            - self.mean_pressure
        ) / self.mean_pressure

        return np.array(
            [
                volume_norm,
                concentration_norm,
                temperature_norm,
                pressure_supply_norm,
                pressure_recirc_norm
            ]
        )

    def switch_to_batch(self):
        self.process.phase = 'batch'
        self.process.work_exchanger_current_volume = self.workexchanger_initial_volume
        self.process.flow_supply_batch = self.process.flow_supply_semibatch
        self.process.flow_supply_batch_previous = self.process.flow_supply_semibatch
        self.process.flow_recirc_batch = self.process.flow_recirc_semibatch
        self.process.flow_recirc_batch_previous = self.process.flow_recirc_semibatch
