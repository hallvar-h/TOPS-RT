import sys
import numpy as np
from synchrophasor.simplePMU import SimplePMU
from .pmu import PMUPublisher


class PMUPublisherCurrentsFreq(PMUPublisher):
    def __init__(self, *args, stations=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stations = stations
        

    @staticmethod
    def get_init_data(rts):
        return [rts.ps.buses, rts.ps.lines['Line'].par, rts.ps.trafos['Trafo'].par]

    def initialize(self, init_data):
        bus_data, line_data, trafo_data = init_data
        all_stations = list(bus_data['name'])        
        if self.stations is None:
            self.stations = all_stations

        channel_names = []
        channel_types = []
        self.bus_masks = []
        self.line_masks_from = []
        self.line_masks_to = []
        self.trafo_masks_from = []
        self.trafo_masks_to = []

        for station_name in self.stations:
            # Get index of station
            station_idx = np.where(np.array(all_stations) == station_name)[0]
            self.bus_masks.append(station_idx)
            
            # Get line currents belonging to station
            line_bus_idx_from = np.where(line_data['from_bus'] == station_name)[0]
            line_bus_idx_to = np.where(line_data['to_bus'] == station_name)[0]
            self.line_masks_from.append(line_bus_idx_from)
            self.line_masks_to.append(line_bus_idx_to)
            
            line_names_from = [f'I[{name}]' for name in line_data['name'][line_bus_idx_from]]
            line_names_to = [f'I[{name}]' for name in line_data['name'][line_bus_idx_to]]

            # Get line currents belonging to station
            trafo_bus_idx_from = np.where(trafo_data['from_bus'] == station_name)[0]
            trafo_bus_idx_to = np.where(trafo_data['to_bus'] == station_name)[0]
            self.trafo_masks_from.append(trafo_bus_idx_from)
            self.trafo_masks_to.append(trafo_bus_idx_to)
            trafo_names_from = [f'I[{name}]' for name in trafo_data['name'][trafo_bus_idx_from]]
            trafo_names_to = [f'I[{name}]r' for name in trafo_data['name'][trafo_bus_idx_to]]
            
            channel_types.append(['v', *['i']*(len(line_names_from) + len(line_names_to) + len(trafo_names_from) + len(trafo_names_to))])
            channel_names.append(['V', *line_names_from, *line_names_to, *trafo_names_from, *trafo_names_to])
            
        # station_names, channel_names = init_data
        self.pmu = SimplePMU(
            self.ip, self.port,
            station_names=self.stations,
            channel_names=channel_names,
            channel_types=channel_types,
            pdc_id=self.pdc_id,
            set_timestamp=False,
            publish_frequency=self.fs,
        )

    @staticmethod
    def read_input_signal(rts):
        # Specify how input signal is read in RealTimeSimulator
        v_full = rts.ps.red_to_full.dot(rts.sol.v)*rts.ps.buses['V_n']*1e3
        line_currents_from = rts.ps.lines['Line'].I_from(rts.sol.x, rts.sol.v)*1e3
        line_currents_to = rts.ps.lines['Line'].I_to(rts.sol.x, rts.sol.v)*1e3
        abs(line_currents_to)
        abs(rts.ps.lines['Line'].i_to(rts.sol.x, rts.sol.v))

        trafo_currents_from = rts.ps.trafos['Trafo'].I_from(rts.sol.x, rts.sol.v)*1e3
        trafo_currents_to = rts.ps.trafos['Trafo'].I_to(rts.sol.x, rts.sol.v)*1e3

        pll_mdl = rts.ps.pll['PLL2']
        freq_est = pll_mdl.freq_est(rts.sol.x, rts.sol.v)
        # pll_mdl.v_measured(rts.sol.x, rts.sol.v)
        # pll_mdl.bus_idx['terminal']
        # rts.sol.v

        return [rts.sol.t, v_full, line_currents_from, line_currents_to, trafo_currents_from, trafo_currents_to, freq_est]

    @staticmethod
    def complex2pol(vec):
        return [(np.abs(vec_), np.angle(vec_)) for vec_ in vec]

    def update(self, input_signal):
        if self.pmu.pmu.clients:  # Check if there is any connected PDCs
            t, v, line_currents_from, line_currents_to, trafo_currents_from, trafo_currents_to, freq_est = input_signal

            time_stamp = round(t * 1e3) * 1e-3

            phasors = []
            freq_data = []
            for mask_v, line_mask_from, line_mask_to, trafo_mask_from, trafo_mask_to in zip(
                self.bus_masks,
                self.line_masks_from,
                self.line_masks_to,
                self.trafo_masks_from,
                self.trafo_masks_to
            ):
                v_pol = self.complex2pol(v[mask_v])
                freq_data.append((freq_est[mask_v[0]] + 1)*50)
                
                line_from_pol = self.complex2pol(line_currents_from[line_mask_from])
                line_to_pol = self.complex2pol(line_currents_to[line_mask_to])

                trafo_from_pol = self.complex2pol(trafo_currents_from[trafo_mask_from])
                trafo_to_pol = self.complex2pol(trafo_currents_to[trafo_mask_to])

                phasors.append([*v_pol, *line_from_pol, *line_to_pol, *trafo_from_pol, *trafo_to_pol])

            # Publish C37.118-snapshot
            self.pmu.publish(time_stamp, phasors, freq_data)