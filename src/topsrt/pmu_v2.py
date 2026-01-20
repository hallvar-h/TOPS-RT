import sys
import numpy as np
from synchrophasor.simplePMU import SimplePMU
from topsrt.pmu import PMUPublisher


class PMUPublisherV2(PMUPublisher):
    def __init__(self, *args, stations=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stations = stations
        

    @staticmethod
    def get_init_data(rts):
        mdl_parameters = {'buses': rts.ps.buses}
        mdl_keys = ['lines', 'trafos', 'gen', 'loads']
        for mdl_key in mdl_keys:
            mdls_dict = getattr(rts.ps, mdl_key)
            mdl_parameters[mdl_key] = {}
            for mdl_subkey, mdl in mdls_dict.items():
                mdl_parameters[mdl_key][mdl_subkey] = mdl.par

        return mdl_parameters  # [rts.ps.buses, rts.ps.lines['Line'].par, rts.ps.trafos['Trafo'].par]

    def initialize(self, init_data):
        # bus_data, line_data, trafo_data = init_data
        mdl_parameters = init_data
        line_data = mdl_parameters['lines']['Line']
        trafo_data = mdl_parameters['trafos']['Trafo']
        # line_data = mdl_parameters['lines']['Line']
        bus_data = mdl_parameters['buses'] 
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
        self.gen_masks = []
        self.load_masks = []

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
            trafo_names_to = [f'I[{name}]' for name in trafo_data['name'][trafo_bus_idx_to]]

            mdl_channel_names = []
            load_masks = {}
            for mdl_key, mdl_data in mdl_parameters['loads'].items():
                units_at_station_idx = np.where(mdl_data['bus'] == station_name)[0]
                load_masks[mdl_key] = units_at_station_idx
                mdl_channel_names += [f'I[{name}]' for name in mdl_data['name'][units_at_station_idx]]
            self.load_masks.append(load_masks)
            load_channel_names = mdl_channel_names
            n_loads = len(load_channel_names)

            mdl_channel_names = []
            gen_masks = {}
            for mdl_key, mdl_data in mdl_parameters['gen'].items():
                units_at_station_idx = np.where(mdl_data['bus'] == station_name)[0]
                gen_masks[mdl_key] = units_at_station_idx
                mdl_channel_names += [f'I[{name}]' for name in mdl_data['name'][units_at_station_idx]]
            self.gen_masks.append(gen_masks)
            gen_channel_names = mdl_channel_names
            n_gen = len(gen_channel_names)

            n_lines_from = len(line_names_from)
            n_lines_to = len(line_names_to)
            n_trafos_from = len(trafo_names_from)
            n_trafos_to = len(trafo_names_to)
            
            channel_types.append(['v', *['i']*(n_lines_from + n_lines_to + n_trafos_from + n_trafos_to + n_loads + n_gen)])
            channel_names.append(['V', *line_names_from, *line_names_to, *trafo_names_from, *trafo_names_to, *load_channel_names, *gen_channel_names])
            
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
        x = rts.sol.x
        v = rts.sol.v

        mdl_keys = ['gen', 'loads']

        # Specify how input signal is read in RealTimeSimulator
        v_full = rts.ps.red_to_full.dot(rts.sol.v)*rts.ps.buses['V_n']*1e3
        line_currents_from = rts.ps.lines['Line'].I_from(rts.sol.x, rts.sol.v)*1e3
        line_currents_to = rts.ps.lines['Line'].I_to(rts.sol.x, rts.sol.v)*1e3
        trafo_currents_from = rts.ps.trafos['Trafo'].I_from(rts.sol.x, rts.sol.v)*1e3
        trafo_currents_to = rts.ps.trafos['Trafo'].I_to(rts.sol.x, rts.sol.v)*1e3

        pll_mdl = rts.ps.pll['PLL2']
        freq_est = pll_mdl.freq_est(rts.sol.x, rts.sol.v)
        # pll_mdl.v_measured(rts.sol.x, rts.sol.v)
        # pll_mdl.bus_idx['terminal']
        # rts.sol.v

        currents = {}
        for mdl_key in mdl_keys:
            mdl_container = getattr(rts.ps, mdl_key)
            currents[mdl_key] = {submdl_key: mdl.I(x, v)*1e3 for submdl_key, mdl in mdl_container.items()}

        return [rts.sol.t, v_full, line_currents_from, line_currents_to, trafo_currents_from, trafo_currents_to, currents, freq_est]

    @staticmethod
    def complex2pol(vec):
        return [(np.abs(vec_), np.angle(vec_)) for vec_ in vec]

    def update(self, input_signal):
        if self.pmu.pmu.clients:  # Check if there is any connected PDCs
            t, v, line_currents_from, line_currents_to, trafo_currents_from, trafo_currents_to, currents, freq_est = input_signal

            time_stamp = round(t * 1e3) * 1e-3

            phasors = []
            freq_data = []
            for mask_v, line_mask_from, line_mask_to, trafo_mask_from, trafo_mask_to, load_mask, gen_mask in zip(
                self.bus_masks,
                self.line_masks_from,
                self.line_masks_to,
                self.trafo_masks_from,
                self.trafo_masks_to,
                self.load_masks,
                self.gen_masks,
            ):
                v_pol = self.complex2pol(v[mask_v])
                freq_data.append((freq_est[mask_v[0]] + 1)*50)
                
                line_from_pol = self.complex2pol(line_currents_from[line_mask_from])
                line_to_pol = self.complex2pol(line_currents_to[line_mask_to])

                trafo_from_pol = self.complex2pol(trafo_currents_from[trafo_mask_from])
                trafo_to_pol = self.complex2pol(trafo_currents_to[trafo_mask_to])
                
                load_currents = np.concatenate([mdl_currents[load_mask[mdl_key]] for mdl_key, mdl_currents in currents['loads'].items()])
                loads_pol = self.complex2pol(load_currents)

                gen_currents = np.concatenate([mdl_currents[gen_mask[mdl_key]] for mdl_key, mdl_currents in currents['gen'].items()])
                gen_pol = self.complex2pol(gen_currents)

                phasors.append([*v_pol, *line_from_pol, *line_to_pol, *trafo_from_pol, *trafo_to_pol, *loads_pol, *gen_pol])

            # Publish C37.118-snapshot
            self.pmu.publish(time_stamp, phasors, freq_data)