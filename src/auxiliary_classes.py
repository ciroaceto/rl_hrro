import numpy as np


class Membrane:
    """Defines the properties of the membrane used in the simulation.
    Units are [m, m, L, m2, L/m2/h/bar, L, -] respectively"""

    def __init__(
        self,
        membrane_name: str = "Ecopro440",
        channel_height: float = 0.000711,
        channel_length: float = 1.0,
        channel_volume: float = 14.50,
        membrane_area: float = 41.0,
        permeability: float = 4.40,
        volume_backflow: float = 5.0,
    ):
        self.membrane_name = membrane_name
        self.channel_height = channel_height
        self.channel_length = channel_length
        self.channel_volume = channel_volume
        self.membrane_area = membrane_area
        self.permeability = permeability
        self.volume_backflow = volume_backflow

    @classmethod
    def membrane_xus180808(cls):
        return cls(
            membrane_name="XUS180808",
            channel_height=0.000864,
            channel_length=1.02,
            channel_volume=13.20,
            membrane_area=30.60,
            permeability=0.72,
            volume_backflow=5.0,
        )

    @classmethod
    def membrane_xus180808_double(cls):
        return cls(
            membrane_name="XUS180808_double",
            channel_height=0.000864,
            channel_length=1.02 * 2,
            channel_volume=13.20 * 2,
            membrane_area=30.60 * 2,
            permeability=0.72,
            volume_backflow=5.0 * 2,
        )


class Solution:
    """Defines the properties of the solution used in the simulation
    Units are [kJ/K kmol, kg/m3, K,-, kg/kmol, kg/m3, Pa s, m2/s] respectively
    Array format allows several solution components to be defined if needed"""

    def __init__(
        self,
        solution_name: str = "Representative NaCl solution for Cr electrolyte",
        gas_constant: float = 8.314,
        # Concentration of 100 g/L represents the full concentration of the electrolyte bath.
        # The factor 10 below is the required concentration factor
        # to restore the rinse to the full concentration.
        # This can vary from 10 for more concentrated rinse
        # to 20 for more dilute rinse water (this factor is subject to random variation)
        concentration_feed=np.array([100 / 10, 0], dtype=float),
        temperature: float = 298.15,
        dissociation=np.array([1.8648, 1.8648], dtype=float),
        relative_molecular_mass=np.array([58.44, 58.44], dtype=float),
        density: float = 1000.0,
        viscosity: float = 0.00089,
        diffusivity=np.array([1.47e-9, 1.47e-9], dtype=float),
        rejection=np.array([0.93, 0.93], dtype=float),
    ):
        self.solution_name = solution_name
        self.gas_constant = gas_constant
        self.concentration_feed = concentration_feed
        self.temperature = temperature
        self.dissociation = dissociation
        self.relative_molecular_mass = relative_molecular_mass
        self.density = density
        self.viscosity = viscosity
        self.diffusivity = diffusivity
        self.rejection = rejection

    def change_concentration_feed(self, concentration):
        self.concentration_feed[0] = concentration / 1000  # convert [mg/L] to [g/L]
        print(self.concentration_feed, concentration, concentration / 1000)
        self.concentration_feed[1] = 0


class DesignParameters:
    """Defines design parameters of the batch RO system.
    Units: volumes [L], diameter [m], pressure [kPa]"""

    def __init__(
        self,
        model: str = "Mark5",
        work_exchanger_initial_volume: float = 69.0,
        volume_pipe_purge: float = 0.114,
        volume_port: float = 0.935,
        volume1: float = 0.8,
        volume2: float = 0.629,
        volume4: float = 0.228,
        supply_pump_eff_semibatch: float = 1.0,
        supply_pump_eff_batch: float = 1.0,
        supply_pump_eff_purge: float = 1.0,
        recirc_pump_eff_semibatch: float = 1.0,
        recirc_pump_eff_batch: float = 1.0,
        recirc_pump_eff_purge: float = 1.0,
        valve_orifice_diameter: float = 0.015,
        coefficient_discharge: float = 0.62,
        delta_pressure_seal: float = 3.5,
    ):
        self.model = model
        self.work_exchanger_initial_volume = work_exchanger_initial_volume
        self.volume1 = volume1
        self.volume2 = volume2
        self.volume4 = volume4
        self.volume_port = volume_port
        self.volume_pipe_purge = volume_pipe_purge
        self.supply_pump_eff_semibatch = supply_pump_eff_semibatch
        self.supply_pump_eff_batch = supply_pump_eff_batch
        self.supply_pump_eff_purge = supply_pump_eff_purge
        self.recirc_pump_eff_semibatch = recirc_pump_eff_semibatch
        self.recirc_pump_eff_batch = recirc_pump_eff_batch
        self.recirc_pump_eff_purge = recirc_pump_eff_purge
        self.valve_orifice_diameter = valve_orifice_diameter
        self.coefficient_discharge = coefficient_discharge
        self.delta_pressure_seal = delta_pressure_seal

    @classmethod
    def Nijhuis_BIA(cls):
        return cls(
            model="Unit installed by Nijhuis at BIA",
            work_exchanger_initial_volume=64.6,
            volume_pipe_purge=0.119,
            volume_port=0.935,
            volume2=1.165,  # the unit is streched by 1 m to allow for 2 modules instead of 1
            supply_pump_eff_semibatch=0.55,
            supply_pump_eff_batch=0.64,
            supply_pump_eff_purge=0.126,
            recirc_pump_eff_semibatch=0.32,
            recirc_pump_eff_batch=0.24,
            recirc_pump_eff_purge=0.476,
            valve_orifice_diameter=0.0254,
        )


class OperationParameters:
    """Define parameters for operating batch RO system
    Units are: flow rates [L/s], times [s]"""

    def __init__(
        self,
        total_cycles: int = 10,
        file_test_number: int = 200,
        flow_rate_semibatch: float = 8 / 60,  # [4, 15] l/min allowed
        flow_rate_batch: float = 8 / 60,  # [4, 15] l/min allowed
        flow_rate_purge: float = 13 / 60,  # [4, 15] l/min allowed
        alpha_semibatch: float = 6,  # min of 1.5; max recirc pump flow rate 70 l/min
        alpha_batch: float = 6,  # as above
        semibatch_time: float = 1800,  # [60, 3600] s (1 min to 1 hour)
        purge_time: float = 0.703,  # [0.5, 1.5] allowed
    ):
        self.total_cycles = total_cycles
        self.file_test_number = file_test_number
        self.flow_rate_semibatch = flow_rate_semibatch
        self.flow_rate_batch = flow_rate_batch
        self.flow_rate_purge = flow_rate_purge
        self.alpha_semibatch = alpha_semibatch
        self.alpha_batch = alpha_batch
        self.semibatch_time = semibatch_time
        self.purge_time = purge_time
