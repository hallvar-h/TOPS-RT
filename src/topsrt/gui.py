from PySide6 import QtWidgets, QtCore
from pyqtgraph.console import ConsoleWidget
import numpy as np


class LineOutageWidget(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Lines')

        lines_per_col = 10
        layout_box = QtWidgets.QGridLayout()

        self.check_boxes = []
        n_lines = self.ps.lines['Line'].n_units
        n_cols = np.ceil(n_lines/lines_per_col)
        for i, line in enumerate(self.ps.lines['Line'].par):
            check_box = QtWidgets.QCheckBox(line['name'])
            check_box.setChecked(True)
            check_box.stateChanged.connect(self.updateLines)
            check_box.setAccessibleName(line['name'])

            row = i%lines_per_col
            col = int((i - row)/lines_per_col)
            
            layout_box.addWidget(check_box, row, col)
            # layout_box.addSpacing(15)
            # layout_box.addSpacing(0)
        

        self.ctrlWidget.setLayout(layout_box)
        self.ctrlWidget.show()

    def updateLines(self):
        if self.sender().isChecked():
            action = 'connect'
        else:
            action = 'disconnect'
        self.ps.lines['Line'].event(self.ps, self.sender().accessibleName(), action)
        # self.ps.network_event('line', self.sender().accessibleName(), action)
        print('t={:.2f}s'.format(self.rts.sol.t) + ': Line ' + self.sender().accessibleName() + ' '+ action + 'ed.')


class DynamicLoadControlWidget(QtWidgets.QWidget):
    def __init__(self, rts, load_type='DynamicLoad', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('DynamicLoadControlWidget')

        layout_box = QtWidgets.QGridLayout()
        self.check_boxes = []
        self.load_mdl = self.ps.loads[load_type]
        self.sliders_G = []
        self.sliders_B = []

        G_0 = self.load_mdl.g_setp(self.ps.x_0, self.ps.v_0)
        B_0 = self.load_mdl.b_setp(self.ps.x_0, self.ps.v_0)
        self.max_Y = max(max(abs(G_0)), max(abs(B_0)))
        max_dev = 2
        
        for i, dyn_load in enumerate(self.load_mdl.par):
            
            # Add text
            label = QtWidgets.QLabel(dyn_load['name'] )
            layout_box.addWidget(label, i, 0)
            
             
            # Add slider, active power (conductance)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
            self.sliders_G.append(slider)
            slider.setMinimum(-100*max_dev)
            slider.setMaximum(100*max_dev)
            slider.valueChanged.connect(lambda state, i=i, target='G': self.updateLoad(i, target))
            slider.setAccessibleName(dyn_load['name'])
            slider.setValue(round(G_0[i]*100/self.max_Y))
            layout_box.addWidget(slider, i, 1)
            
            # Add slider, reactive power (susceptance)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
            self.sliders_B.append(slider)
            slider.setMinimum(-100*max_dev)
            slider.setMaximum(100*max_dev)
            slider.valueChanged.connect(lambda state, i=i, target='B': self.updateLoad(i, target))
            slider.setAccessibleName(dyn_load['name'])
            slider.setValue(round(B_0[i]*100/self.max_Y))
            layout_box.addWidget(slider, i, 2)
            

        self.ctrlWidget.setLayout(layout_box)
        self.ctrlWidget.show()

    def updateLoad(self, load_idx, target):
        # print(load_idx, target, len(self.sliders_G), len(self.sliders_B))
        if target == 'G':
            self.load_mdl.set_input('g_setp', self.sliders_G[load_idx].value()*self.max_Y/100, load_idx)
        else:
            self.load_mdl.set_input('b_setp', self.sliders_B[load_idx].value()*self.max_Y/100, load_idx)


class VSCControlWidget(QtWidgets.QWidget):
    def __init__(self, rts, max_dev=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('VSCControlWidget')

        layout_box = QtWidgets.QGridLayout()
        self.check_boxes = []
        self.target_mdl = self.ps.vsc['VSC']
        self.active_power_setp_fun = 'P_setp'
        self.reactive_power_setp_fun = 'Q_setp'

        self.sliders_active_power = []
        self.sliders_reactive_power = []

        active_power_0 = getattr(self.target_mdl, self.active_power_setp_fun)(self.ps.x_0, self.ps.v_0)
        reactive_power_0 = getattr(self.target_mdl, self.reactive_power_setp_fun)(self.ps.x_0, self.ps.v_0)
        self.max_S = max(max(abs(active_power_0)), max(abs(reactive_power_0)))
        
        for i, mdl in enumerate(self.target_mdl.par):
            
            # Add text
            label = QtWidgets.QLabel(mdl['name'] )
            layout_box.addWidget(label, i, 0)
            
            # Add slider, active power (conductance)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
            self.sliders_active_power.append(slider)
            slider.setMinimum(-100*max_dev)
            slider.setMaximum(100*max_dev)
            slider.valueChanged.connect(lambda state, i=i, target='active': self.updateLoad(i, target))
            slider.setAccessibleName(mdl['name'])
            slider.setValue(round(active_power_0[i]*100/self.max_S))
            layout_box.addWidget(slider, i, 1)
            
            # Add slider, reactive power (susceptance)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
            self.sliders_reactive_power.append(slider)
            slider.setMinimum(-100*max_dev)
            slider.setMaximum(100*max_dev)
            slider.valueChanged.connect(lambda state, i=i, target='reactive': self.updateLoad(i, target))
            slider.setAccessibleName(mdl['name'])
            slider.setValue(round(reactive_power_0[i]*100/self.max_S))
            layout_box.addWidget(slider, i, 2)
            

        self.ctrlWidget.setLayout(layout_box)
        self.ctrlWidget.show()

    def updateLoad(self, idx, target):
        # print(load_idx, target, len(self.sliders_G), len(self.sliders_B))
        if target == 'active':
            self.target_mdl.set_input(self.active_power_setp_fun, self.sliders_active_power[idx].value()*self.max_S/100, idx)
        else:
            self.target_mdl.set_input(self.reactive_power_setp_fun, self.sliders_reactive_power[idx].value()*self.max_S/100, idx)


class SimulationControl(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.setBackgroundRole(QtGui.QPalette.Base)
        # self.setAutoFillBackground(True)

        self.rts = rts
        self.ps = rts.ps

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Simulation Controls')
        layout = QtWidgets.QGridLayout()

        # Add speed slider
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
        self.speed_slider = slider
        slider.setMinimum(1)
        slider.setMaximum(200)
        slider.valueChanged.connect(lambda state: self.updateSpeed())
        slider.setAccessibleName('Simulation speed')
        slider.setValue(100)
        layout.addWidget(slider, 0, 0)

        # Pause button
        button = QtWidgets.QPushButton('Pause')
        button.setCheckable(True)
        button.setChecked(False)
        layout.addWidget(button, 1, 0)
        button.clicked.connect(lambda state: self.pauseSimulation())

        # Reset button
        button = QtWidgets.QPushButton('Reset')
        button.setCheckable(False)
        layout.addWidget(button, 2, 0)
        button.clicked.connect(lambda state: self.resetSimulation())

        self.ctrlWidget.setLayout(layout)
        self.ctrlWidget.show()

    def updateSpeed(self):
        self.rts.speed = self.speed_slider.value()/100

    def resetSimulation(self):
        self.rts.toggle_pause()
        # self.ps.init_dyn_sim()
        self.rts.sol.x[:] = self.ps.x_0
        self.rts.sol.v[:] = self.ps.v_0
        self.rts.toggle_pause()

    def pauseSimulation(self):
        self.rts.toggle_pause()


