import numpy as np
from typing import Type
from auxiliary_classes import (
    Membrane,
    Solution,
    DesignParameters,
    OperationParameters,
)


class Step_HRRO:
    """A class used to simulate a batch/semibatch RO system"""

    def __init__(
        self,
        membrane: Type[Membrane],
        solution: Type[Solution],
        design: Type[DesignParameters],
        operation: Type[OperationParameters],
    ):
        # TIME ATTRIBUTES [s]:
        self.time_start = 0  # start cycle time
        self.time_step = 0.2
        self.time = self.time_start
        self.time_agent_step = 60 - self.time_step / 2  # frequency of action decision
        self.time_agent_last = self.time_start  # time of last action decision

        # OPERATION ATTRIBUTES:
        self.total_cycles = operation.total_cycles
        self.current_cycle: int = 1
        self.file_test_number = operation.file_test_number
        self.flow_supply_semibatch = operation.flow_rate_semibatch
        self.flow_supply_semibatch_previous = operation.flow_rate_semibatch
        self.flow_supply_semibatch_setpoint = operation.flow_rate_semibatch
        self.flow_supply_batch = operation.flow_rate_batch
        self.flow_supply_batch_previous = operation.flow_rate_batch
        self.flow_supply_batch_setpoint = operation.flow_rate_batch
        self.flow_supply_purge = operation.flow_rate_purge
        self.flow_supply_purge_previous = operation.flow_rate_purge
        self.flow_supply_purge_setpoint = operation.flow_rate_purge
        self.alpha_semibatch = operation.alpha_semibatch
        self.alpha_batch = operation.alpha_batch
        self.flow_recirc_semibatch = self.alpha_semibatch * self.flow_supply_semibatch
        self.flow_recirc_semibatch_previous = (
            self.alpha_semibatch * self.flow_supply_semibatch
        )
        self.flow_recirc_semibatch_setpoint = (
            self.alpha_semibatch * self.flow_supply_semibatch
        )
        self.flow_recirc_batch = self.alpha_batch * self.flow_supply_batch
        self.flow_recirc_batch_previous = self.alpha_batch * self.flow_supply_batch
        self.flow_recirc_batch_setpoint = self.alpha_batch * self.flow_supply_batch
        self.semibatch_time = operation.semibatch_time  # only used for testing purposes
        self.semibatch_current_time = 0
        self.purge_time = operation.purge_time

        # SOLUTION ATTRIBUTES:
        self.solution_name = solution.solution_name
        self.concentration_feed = solution.concentration_feed  # [kg/m3]
        self.temperature = solution.temperature  # [K]
        self.dissociation = solution.dissociation
        self.gas_constant = solution.gas_constant  # [kJ/K kmol]
        self.relative_molecular_mass = solution.relative_molecular_mass  # [kg/kmol]
        self.density = solution.density  # [kg/m3]
        self.diffusivity = solution.diffusivity  # [m2/s]
        self.viscosity = solution.viscosity  # [Pa s]
        self.rejection = solution.rejection

        # MEMBRANE ATTRIBUTES:
        self.membrane_name = membrane.membrane_name
        self.channel_height = membrane.channel_height  # [m]
        self.channel_length = membrane.channel_length  # [m]
        self.channel_volume = membrane.channel_volume  # [L]
        self.membrane_area = membrane.membrane_area  # [m2]
        self.permeability = membrane.permeability / 3.6e8  # [LMH/bar] to [m/s/kPa]
        self.volume_backflow = membrane.volume_backflow  # [L]

        # DESIGN ATTRIBUTES [all volumes in L]:
        self.model = design.model
        self.work_exchanger_initial_volume = design.work_exchanger_initial_volume
        self.work_exchanger_current_volume = self.work_exchanger_initial_volume
        self.feed_tank_current_volume = 10_000
        # RO module is divided into 25 cells based on MSc work of B.Mattu
        self.cell_volume = self.channel_volume / 25
        self.cell_area = self.membrane_area / 25

        # Minor pipe and port volumes:
        # pipe upstream of recirc pump:
        self.volume1 = design.volume1
        # recirc pump internal volume, pipe downstream, and piston internal volume:
        self.volume2 = design.volume2
        # note: volume 3 corresponds to work exchanger volume (already defined above)
        # pipe connecting work exchanger to RO module:
        self.volume4 = design.volume4
        # RO module ports (inlet and outlet respectively):
        self.volume5 = design.volume_port
        self.volume31 = design.volume_port
        # purgeable pipe volume:
        self.volume32 = design.volume_pipe_purge

        # Purgeable internal volume
        self.volume_purge = (
            self.channel_volume + self.volume5 + self.volume31 + self.volume32
        )
        self.update_volume_brine_flow_recirc()

        self.pressure_supply = 0
        self.pressure_recirc = 0

        # PUMP ATTRIBUTES
        self.spump_flowrate_delta = 0.2 / 60   # increase-decrease L/s^2 time_step 0.2
        self.rpump_flowrate_delta = 0.4 / 60   # increase-decrease L/s^2 time_step 0.2

        # Supply and recirculation pump efficiencies:
        self.spump_eff_semibatch = design.supply_pump_eff_semibatch
        self.spump_eff_batch = design.supply_pump_eff_batch
        self.spump_eff_purge = design.supply_pump_eff_purge
        self.rpump_eff_semibatch = design.recirc_pump_eff_semibatch
        self.rpump_eff_batch = design.recirc_pump_eff_batch
        self.rpump_eff_purge = design.recirc_pump_eff_purge

        # Valves and piston parameters [-, m, kPa]:
        self.coefficient_discharge = design.coefficient_discharge
        self.valve_orifice_diameter = design.valve_orifice_diameter
        self.delta_pressure_seal = design.delta_pressure_seal

        # Initialise concentration as the feed concentration throughout:
        self.reset_concentrations()

        # Initialise values of flux [m/s] for each cell as uniform over module:
        self.flux = (
            0.001 * np.ones(25) * self.flow_supply_semibatch / self.membrane_area
        )

        # Initialise RO module pressure at arbitrary very small value to
        # start system [kPa]:
        # self.transmembrane_pressure = 0.1
        # Initialise osmotic pressure:
        # self.osmotic_pressure = self.calculate_average_osmotic_pressure()

        # Zero water and energy meters [L, kJ]:
        self.initialize_volume_energy_semibatchtime()

        # Initial phase
        self.phase: str = 'semibatch'

        # Initialise matrix for purge and refill phase
        self.matrix_purge = self.generate_matrix_purge()

        # For test purposes only
        self.test = ''

    def set_agent_decisions(
            self,
            flow_supply: float = None,
            flow_recirc: float = None,
            purge_time: float = None
    ):
        if self.phase == 'semibatch':
            self.flow_supply_semibatch_setpoint = flow_supply
            self.flow_recirc_semibatch_setpoint = flow_recirc
        elif self.phase == 'batch':
            self.flow_supply_batch_setpoint = flow_supply
            self.flow_recirc_batch_setpoint = flow_recirc
        else:
            self.flow_supply_purge_setpoint = flow_supply
            self.purge_time = purge_time
            self.update_volume_brine_flow_recirc()

    def update_volume_brine_flow_recirc(self):
        self.volume_brine = self.volume_purge * self.purge_time
        self.flow_recirc_refill = self.work_exchanger_initial_volume / (
            self.volume_brine / self.flow_supply_purge
        )

    def calculate_permeate_time_cycle(self):
        return (self.time - self.time_start) / 60

    def calculate_permeate_flow_rate(self):
        return (
            self.permeate_volume
            / self.calculate_permeate_time_cycle()
        )

    def reset_concentrations(self):
        self.concentrations = (
            np.ones([33, np.size(self.concentration_feed)]) * self.concentration_feed
        )

    def check_alpha(self):
        if self.phase == 'semibatch':
            return self.flow_recirc_semibatch / self.flow_supply_semibatch
        elif self.phase == 'batch':
            return self.flow_recirc_batch / self.flow_supply_batch
        else:
            return 2

    def generate_matrix_semibatch(self, flow_supply, flow_recirc):
        """Create matrix for semi-batch phase. The matrix relates the rate of change
        of concentration to the current concentration for each phase of the process
        (i.e., system of ODEs) according to the mass balance for each volume element.

        Each section is fed by upstream section and discharges to downstream
        section (e.g., volume1 pipe section is fed by supply pump and volume
        32 pipe section)

        Inside the RO module, the in and outflows gradually decrease due to
        permeation.
        """
        # Initialise values of flow at inlet of each cell, and outlet of last cell:
        flow_total = flow_supply + flow_recirc
        self.flow_cell = np.ones(26) * flow_total
        for i in range(1, 26):
            # subtract permeation
            self.flow_cell[i] = (
                self.flow_cell[i - 1] - 1000 * self.cell_area * self.flux[i - 1]
            )
        matrix = np.zeros([33, 33])
        matrix[1, 0] = flow_supply / self.volume1
        matrix[1, 1] = -flow_total / self.volume1
        matrix[1, 32] = flow_recirc / self.volume1
        matrix[2, 1] = flow_total / self.volume2
        matrix[2, 2] = -flow_total / self.volume2
        matrix[3, 2] = flow_total / self.work_exchanger_initial_volume
        matrix[3, 3] = -flow_total / self.work_exchanger_initial_volume
        matrix[4, 3] = flow_total / self.volume4
        matrix[4, 4] = -flow_total / self.volume4
        matrix[5, 4] = flow_total / self.volume5
        matrix[5, 5] = -flow_total / self.volume5
        for i in range(6, 31):
            matrix[i, i - 1] = self.flow_cell[i - 6] / self.cell_volume
            matrix[i, i] = -self.flow_cell[i - 5] / self.cell_volume
        matrix[31, 30] = flow_recirc / self.volume31
        matrix[31, 31] = -flow_recirc / self.volume31
        matrix[32, 31] = flow_recirc / self.volume32
        matrix[32, 32] = -flow_recirc / self.volume32
        return matrix

    def generate_matrix_batch(self, flow_supply, flow_recirc):
        """Create elements of matrix for batch phase.
        Similar to semibatch matrix but change a few elements"""
        matrix = self.generate_matrix_semibatch(flow_supply, flow_recirc)
        matrix[1, 0] = 0
        matrix[1, 1] = -flow_recirc / self.volume1
        matrix[2, 1] = flow_recirc / self.volume2
        matrix[2, 2] = -flow_recirc / self.volume2
        return matrix

    def generate_matrix_purge(self):
        """Create elements of matrix for purge-and-refill phase"""
        matrix = np.zeros([33, 33])
        for i in range(5, 31):
            matrix[i, i] = -self.flow_supply_purge / self.cell_volume
            matrix[i, i + 1] = self.flow_supply_purge / self.cell_volume
        matrix[31, 31] = -self.flow_supply_purge / self.volume31
        matrix[31, 32] = self.flow_supply_purge / self.volume31
        matrix[32, 0] = self.flow_supply_purge / self.volume32
        matrix[32, 32] = -self.flow_supply_purge / self.volume32
        return matrix

    def calculate_concentration_dash(self, concentrations, phase):
        """Relate rate of change of concentraton to current concentration using matrix
        (system of ODEs)"""
        if phase == 'semibatch':
            matrix = self.generate_matrix_semibatch(
                self.flow_supply_semibatch, self.flow_recirc_semibatch
            )
            concentration_dash = matrix @ concentrations
        if phase == 'batch':
            # Two elements depend on the current volume
            # they represent the flow in and out of the W.E.
            matrix = self.generate_matrix_batch(
                self.flow_supply_batch, self.flow_recirc_batch
            )
            matrix[3, 2] = self.flow_recirc_batch / self.work_exchanger_current_volume
            matrix[3, 3] = -self.flow_recirc_batch / self.work_exchanger_current_volume
            concentration_dash = matrix @ concentrations
        if phase == 'purge&refill':
            self.matrix_purge = self.generate_matrix_purge()
            concentration_dash = self.matrix_purge @ concentrations
        return concentration_dash

    def apply_euler(self):
        """Apply improved Euler method to predict concentration at next time step.
        First, it estimates rate of change based on current value. Then, the rate
        of change is estimated after half time step, improving the first estimation.
        The new estimation is used to predict the concentration after the full
        time step.
        """
        concentration_dash = self.calculate_concentration_dash(
            self.concentrations, self.phase
        )
        concentration_estimate = (
            self.concentrations + concentration_dash * self.time_step / 2
        )
        concentration_dash = self.calculate_concentration_dash(
            concentration_estimate, self.phase
        )
        self.concentrations += concentration_dash * self.time_step

    def apply_RK4(self):
        """Apply Runge-Kutta 4th order method to predict concentration at next time step
        (alternative to Euler method that is expected to have lower error)
        """
        k1 = self.time_step * self.calculate_concentration_dash(
            self.concentrations, self.phase
        )
        k2 = self.time_step * self.calculate_concentration_dash(
            self.concentrations + k1 / 2, self.phase
        )
        k3 = self.time_step * self.calculate_concentration_dash(
            self.concentrations + k2 / 2, self.phase
        )
        k4 = self.time_step * self.calculate_concentration_dash(
            self.concentrations + k3, self.phase
        )
        self.concentrations += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def vantHoff(self, concentration):
        """van't Hoff formula, osmotic pressure calculation"""
        return (
            concentration
            * self.temperature
            * self.gas_constant
            * self.dissociation
            / self.relative_molecular_mass
        )

    def calculate_average_osmotic_pressure(self):
        """
        Based on mean of concentration in RO module with help of van't Hoff formula
        """
        concentration_avg = self.concentrations[6:31, :].mean(axis=0)
        return self.vantHoff(concentration_avg)

    def calculate_flux(self, supply_flow):
        """
        Calculate transmembrane water flux at each cell, using
        solution diffusion equation, summing osmotic pressure contributions
        from each solute
        """
        self.average_flux = 0.001 * supply_flow / self.membrane_area
        # Calculate transmembrane pressure based on average flux and average osmotic
        # pressures along module. For multiple solutes, osmotic contributions are summed
        self.transmembrane_pressure = (
            self.rejection * self.Cp * self.osmotic_pressure
        ).sum() + self.average_flux / self.permeability
        local_osmotic_pressure = self.vantHoff(self.concentrations[6:31, :])
        flux = self.permeability * (
            self.transmembrane_pressure
            - (self.rejection * self.Cp * local_osmotic_pressure).sum(axis=1)
        )
        # adjust flux to ensure consistency with average flux as determined
        # by feed flow.
        flux *= self.average_flux / flux.mean()
        return flux

    def calculate_concentration_polarisation(
        self,
        flow_recirc,
        flow_supply,
        channel_height,
        channel_length,
        density,
        viscosity,
        diffusivity,
        membrane_area,
    ):
        """Calculate concentration polarisation using
        Sherwood correlation."""
        width = membrane_area / channel_length
        velocity_crossflow = (
            0.001 * (flow_recirc + 0.5 * flow_supply) / (0.5 * channel_height * width)
        )
        Reynolds = density * velocity_crossflow * 0.5 * channel_height / viscosity
        Schmidt = viscosity / (density * diffusivity)
        Sherwood = 0.2 * (Reynolds**0.57) * (Schmidt**0.4)
        mass_transfer_coefficient = Sherwood * diffusivity / (0.5 * channel_height)
        average_flux = 0.001 * flow_supply / membrane_area
        return np.real(np.exp(average_flux / mass_transfer_coefficient))

    def calculate_crossflow_pressure_drop(
        self, flow_rate, channel_height, channel_length, membrane_area
    ):
        """Calculate crossflow pressure drop based on Haidari et al"""
        width = membrane_area / channel_length
        velocity_crossflow = 0.001 * flow_rate / (0.5 * channel_height * width)
        return 791 * channel_length * velocity_crossflow**1.63

    def calculate_valve_pressure_drop(
        self, flow, valve_orifice_diameter, coefficient_discharge, density
    ):
        """Calculate valve pressure drop [kPa] based on Torricelli equation"""
        velocity = 0.001 * flow / (0.25 * 3.1412 * valve_orifice_diameter**2)
        return 0.001 * (density / 2) * (velocity / coefficient_discharge) ** 2

    def initialize_volume_energy_semibatchtime(self):
        """Initialize to zero water and energy meters"""
        self.supply_volume = 0
        self.brine_collected = 0
        self.permeate_volume = 0
        self.recovery = 0
        self.supply_energy_semibatch = 0
        self.supply_energy_batch = 0
        self.supply_energy_purge = 0
        self.recirc_energy_semibatch = 0
        self.recirc_energy_batch = 0
        self.recirc_energy_purge = 0
        self.semibatch_current_time = 0

    def calculate_SEC(self):
        """
        Calculate specific energy consumptions at end of cycle, converting
        from [kJ/L] to [kWh/m3]
        """
        self.SECsupply_sbatch = (
            self.supply_energy_semibatch / 3.6 / self.permeate_volume
        )
        self.SECrecirc_sbatch = (
            self.recirc_energy_semibatch / 3.6 / self.permeate_volume
        )
        self.SECsupply_batch = (self.supply_energy_batch / 3.6) / self.permeate_volume
        self.SECrecirc_batch = (self.recirc_energy_batch / 3.6) / self.permeate_volume
        self.SECsupply_purge = (self.supply_energy_purge / 3.6) / self.permeate_volume
        self.SECrecirc_purge = (self.recirc_energy_purge / 3.6) / self.permeate_volume
        self.SECsupply = (
            self.SECsupply_sbatch + self.SECsupply_batch + self.SECsupply_purge
        )
        self.SECrecirc = (
            self.SECrecirc_sbatch + self.SECrecirc_batch + self.SECrecirc_purge
        )
        self.SEC_Total = self.SECsupply + self.SECrecirc
        self.SEC_Total_current = self.SEC_Total

    def update_flowrate(self, setpoint, previous, delta):
        if previous == setpoint:
            return setpoint, setpoint

        elif previous < setpoint:
            previous += delta
            if previous >= setpoint:
                return setpoint, min(setpoint, previous - delta / 2)

            else:
                return previous, previous - delta / 2

        else:
            previous -= delta
            if previous <= setpoint:
                return setpoint, max(setpoint, previous + delta / 2)

            else:
                return previous, previous + delta / 2

    def update_pump_flowrates(self):
        if self.phase == 'semibatch':
            (
                self.flow_supply_semibatch_previous,
                self.flow_supply_semibatch,
            ) = self.update_flowrate(
                    self.flow_supply_semibatch_setpoint,
                    self.flow_supply_semibatch_previous,
                    self.spump_flowrate_delta,
            )

            (
                self.flow_recirc_semibatch_previous,
                self.flow_recirc_semibatch,
            ) = self.update_flowrate(
                    self.flow_recirc_semibatch_setpoint,
                    self.flow_recirc_semibatch_previous,
                    self.rpump_flowrate_delta,
            )

        elif self.phase == 'batch':
            (
                self.flow_supply_batch_previous,
                self.flow_supply_batch,
            ) = self.update_flowrate(
                    self.flow_supply_batch_setpoint,
                    self.flow_supply_batch_previous,
                    self.spump_flowrate_delta,
            )

            (
                self.flow_recirc_batch_previous,
                self.flow_recirc_batch,
            ) = self.update_flowrate(
                    self.flow_recirc_batch_setpoint,
                    self.flow_recirc_batch_previous,
                    self.rpump_flowrate_delta,
            )

        else:
            (
                self.flow_supply_purge_previous,
                self.flow_supply_purge
            ) = self.update_flowrate(
                    self.flow_supply_purge_setpoint,
                    self.flow_supply_purge_previous,
                    self.spump_flowrate_delta,
            )

    def calculate_semibatch_pressure_drop_concentration_polarization(self):

        # pressure drop across bypass valve:
        pressure_drop_valve1 = self.calculate_valve_pressure_drop(
            self.flow_supply_semibatch,
            self.valve_orifice_diameter,
            self.coefficient_discharge,
            self.density,
        )
        # pressure drop across recirc valve:
        pressure_drop_valve2 = self.calculate_valve_pressure_drop(
            self.flow_recirc_semibatch + self.flow_supply_semibatch,
            self.valve_orifice_diameter,
            self.coefficient_discharge,
            self.density,
        )
        pressure_drop_module = self.calculate_crossflow_pressure_drop(
            self.flow_recirc_semibatch + 0.5 * self.flow_supply_semibatch,
            self.channel_height,
            self.channel_length,
            self.membrane_area,
        )
        self.Cp = self.calculate_concentration_polarisation(
            self.flow_recirc_semibatch,
            self.flow_supply_semibatch,
            self.channel_height,
            self.channel_length,
            self.density,
            self.viscosity,
            self.diffusivity,
            self.membrane_area,
        )

        return pressure_drop_valve1, pressure_drop_valve2, pressure_drop_module

    def run_semibatch(self):
        """Run the simulation of the semibatch phase of the cycle
        The cycle starts with the semibatch phase. Semibatch phase
        continues until duration is exceeded."""

        (
            pressure_drop_valve1,
            pressure_drop_valve2,
            pressure_drop_module
        ) = self.calculate_semibatch_pressure_drop_concentration_polarization()

        while self.time - self.time_agent_last < self.time_agent_step:

            self.semibatch_current_time += self.time_step
            self.time += self.time_step

            if not (
                self.flow_supply_semibatch == self.flow_supply_semibatch_setpoint
                and self.flow_recirc_semibatch == self.flow_recirc_semibatch_setpoint
            ):
                self.update_pump_flowrates()

                (
                    pressure_drop_valve1,
                    pressure_drop_valve2,
                    pressure_drop_module
                ) = self.calculate_semibatch_pressure_drop_concentration_polarization()

            self.apply_euler()
            self.feed_tank_current_volume -= self.time_step * self.flow_supply_semibatch
            self.supply_volume += self.time_step * self.flow_supply_semibatch
            self.permeate_volume += self.time_step * self.flow_supply_semibatch
            self.osmotic_pressure = self.calculate_average_osmotic_pressure()
            self.flux = self.calculate_flux(self.flow_supply_semibatch)
            self.pressure_supply = (
                self.transmembrane_pressure
                - 0.5 * pressure_drop_module
                + pressure_drop_valve1
            )
            self.pressure_recirc = pressure_drop_module + pressure_drop_valve2
            self.supply_energy_semibatch += (
                self.time_step
                * self.flow_supply_semibatch
                * 0.001
                * self.pressure_supply
                / self.spump_eff_semibatch
            )
            self.recirc_energy_semibatch += (
                self.time_step
                * (self.flow_supply_semibatch + self.flow_recirc_semibatch)
                * 0.001
                * self.pressure_recirc
                / self.rpump_eff_semibatch
            )

            if self.test == 'yes':
                if self.semibatch_current_time > self.semibatch_time:
                    break

            if (
                self.feed_tank_current_volume <= 250 or
                self.pressure_supply / 100 >= 120
            ):
                break

        self.time_agent_last = self.time

    def calculate_batch_pressure_drop_concentration_polarization(self):

        # pressure drop across recirc valve:
        pressure_drop_valve2 = self.calculate_valve_pressure_drop(
            self.flow_recirc_batch + self.flow_supply_batch,
            self.valve_orifice_diameter,
            self.coefficient_discharge,
            self.density,
        )
        pressure_drop_module = self.calculate_crossflow_pressure_drop(
            self.flow_recirc_batch + 0.5 * self.flow_supply_batch,
            self.channel_height,
            self.channel_length,
            self.membrane_area,
        )
        self.Cp = self.calculate_concentration_polarisation(
            self.flow_recirc_batch,
            self.flow_supply_batch,
            self.channel_height,
            self.channel_length,
            self.density,
            self.viscosity,
            self.diffusivity,
            self.membrane_area,
        )

        return pressure_drop_valve2, pressure_drop_module

    def run_batch(self):
        """
        Run the simulation of the batch phase of the cycle
        The batch phase continues as long as the WE vessel volume is positive.
        The value of work_exchanger_current_volume at midincrement is used for
        accurate calculation.
        """

        (
            pressure_drop_valve2,
            pressure_drop_module
        ) = self.calculate_batch_pressure_drop_concentration_polarization()

        while self.time - self.time_agent_last < self.time_agent_step:

            if not (
                self.flow_supply_batch == self.flow_supply_batch_setpoint
                and self.flow_recirc_batch == self.flow_recirc_batch_setpoint
            ):
                self.update_pump_flowrates()

                (
                    pressure_drop_valve2,
                    pressure_drop_module
                ) = self.calculate_batch_pressure_drop_concentration_polarization()

            self.work_exchanger_current_volume -= (
                self.flow_supply_batch * self.time_step / 2
            )
            if (
                (
                    self.work_exchanger_current_volume
                    - self.flow_supply_batch * self.time_step / 2
                 ) < 0.1
            ):
                self.phase = 'purge&refill'
                break
            self.apply_euler()
            self.time += self.time_step
            # the final value of V after full time step:
            self.work_exchanger_current_volume -= (
                self.flow_supply_batch * self.time_step / 2
            )
            self.feed_tank_current_volume -= self.time_step * self.flow_supply_batch
            self.supply_volume += self.time_step * self.flow_supply_batch
            self.permeate_volume += self.time_step * self.flow_supply_batch
            self.osmotic_pressure = self.calculate_average_osmotic_pressure()
            self.flux = self.calculate_flux(self.flow_supply_batch)
            self.pressure_supply = (
                self.transmembrane_pressure
                + 0.5 * pressure_drop_module
                + pressure_drop_valve2
                + self.delta_pressure_seal
            )
            self.pressure_recirc = pressure_drop_module + pressure_drop_valve2
            self.supply_energy_batch += (
                self.time_step
                * self.flow_supply_batch
                * 0.001
                * self.pressure_supply
                / self.spump_eff_batch
            )
            self.recirc_energy_batch += (
                self.time_step
                * self.flow_recirc_batch
                * 0.001
                * self.pressure_recirc
                / self.rpump_eff_batch
            )

            if (
                self.feed_tank_current_volume <= 250 or
                self.pressure_supply / 100 >= 120
            ):
                self.pressure_supply = 124
                break

        self.time_agent_last = self.time

    def calculate_purge_pressure_drop(self):

        # pressure drop across bypass valve:
        pressure_drop_valve1 = self.calculate_valve_pressure_drop(
            self.flow_recirc_refill + self.flow_supply_purge,
            self.valve_orifice_diameter,
            self.coefficient_discharge,
            self.density,
        )
        # pressure drop across brine valve
        pressure_drop_valve3 = self.calculate_valve_pressure_drop(
            self.flow_supply_purge,
            self.valve_orifice_diameter,
            self.coefficient_discharge,
            self.density,
        )

        return pressure_drop_valve1, pressure_drop_valve3

    def run_purge_refill(self):
        """Run the simulation of the purge and refill phase of the cycle

        The unpurged pipe sections now get purged by the feed. The work
        exchanger concentration and volume gets reset to initial state
        with a correction for salt retention in unpurged pipe sections.

        Purge until the specified amont of brine is collected: volume_brine.

        At the end of purge and refill phase backflow occurs.

        Energy is converted to KWh and SEC calculated
        """

        (
            pressure_drop_valve1,
            pressure_drop_valve3
        ) = self.calculate_purge_pressure_drop()

        self.concentrations[1, :] = self.concentrations[0, :]
        self.concentrations[2, :] = self.concentrations[0, :]
        self.concentrations[3, :] = (
            self.concentrations[0, :]
            + (self.volume1 / self.work_exchanger_initial_volume)
            * (self.concentrations[1, :] - self.concentrations[0, :])
            + (self.volume2 / self.work_exchanger_initial_volume)
            * (self.concentrations[2, :] - self.concentrations[0, :])
        )

        self.work_exchanger_current_volume = self.work_exchanger_initial_volume

        while self.brine_collected < self.volume_brine:

            if not self.flow_supply_purge == self.flow_supply_purge_setpoint:
                self.update_pump_flowrates()

                (
                    pressure_drop_valve1,
                    pressure_drop_valve3
                ) = self.calculate_purge_pressure_drop()

            self.apply_euler()
            self.time += self.time_step
            self.feed_tank_current_volume -= self.time_step * self.flow_supply_purge
            self.supply_volume += self.time_step * self.flow_supply_purge
            self.brine_collected += self.time_step * self.flow_supply_purge
            pressure_drop_module = self.calculate_crossflow_pressure_drop(
                self.flow_supply_purge,
                self.channel_height,
                self.channel_length,
                self.membrane_area,
            )
            # calculate supply and recirc pump energy usages:
            # may need to add venturi effect
            self.pressure_supply = (
                pressure_drop_valve1 + pressure_drop_module + pressure_drop_valve3
            )
            self.pressure_recirc = self.delta_pressure_seal + pressure_drop_valve1
            self.supply_energy_purge += (
                self.time_step
                * self.flow_supply_purge
                * 0.001
                * self.pressure_supply
                / self.spump_eff_purge
            )
            self.recirc_energy_purge += (
                self.time_step
                * self.flow_recirc_refill
                * 0.001
                * self.pressure_recirc
                / self.rpump_eff_purge
            )
            self.osmotic_pressure = self.calculate_average_osmotic_pressure()

            if self.feed_tank_current_volume <= 250:
                break

        self.time_agent_last = self.time
        self.time_start = self.time
        self.calculate_recovery()
        self.current_cycle += 1
        self.phase = 'semibatch'

    def calculate_recovery(self):
        self.brine_collected += self.volume_backflow
        self.permeate_volume -= self.volume_backflow
        self.recovery = self.permeate_volume / self.supply_volume

    def step(self):
        if self.phase == 'semibatch':
            return self.run_semibatch()
        elif self.phase == 'batch':
            return self.run_batch()
        else:
            return self.run_purge_refill()

    def simulate(self):
        """Testing the simulation of total_cycles of the system"""
        self.test = 'yes'
        while self.current_cycle <= self.total_cycles:
            self.time_start = self.time
            self.initialize_volume_energy_semibatchtime()
            while self.time - self.time_start <= self.semibatch_time:
                self.run_semibatch()
            self.phase = 'batch'
            while self.work_exchanger_current_volume > 0:
                self.run_batch()
            self.run_purge_refill()
            self.calculate_recovery()
            self.calculate_SEC()
            self.current_cycle += 1
        return [
            self.SECsupply_sbatch,
            self.SECsupply_batch,
            self.SECsupply_purge,
            self.SECsupply,
            self.SECrecirc_sbatch,
            self.SECrecirc_batch,
            self.SECrecirc_purge,
            self.SECrecirc,
            self.SEC_Total,
        ]
