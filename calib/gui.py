#!/usr/bin/env python

import sys
import os
import subprocess
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

import process_darks
import gen_constants

class CalibGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(CalibGUI, self).__init__()
        self.procdk = process_darks.ProcessDarks()
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle('AGIPD Calibration GUI')
        self.setGeometry(100, 100, 1000, 800)
        #self.setFixedSize(1000, 800)
        overall = QtWidgets.QWidget()
        self.setCentralWidget(overall)
        layout = QtWidgets.QVBoxLayout(overall)
        layout.setContentsMargins(0, 0, 0, 0)

        self._init_menubar()
        options_widget = self._init_optionsarea()

        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.plot = pg.PlotWidget(self)
        plot_splitter.addWidget(self.plot)
        self.imview = pg.ImageView(self)
        plot_splitter.addWidget(self.imview)
        plot_splitter.setSizes([400, 400])
        self.genct = gen_constants.GenConstants(plotter=self.plot, imview=self.imview)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(main_splitter)
        main_splitter.addWidget(plot_splitter)
        main_splitter.addWidget(options_widget)

        self.show()

    def _init_menubar(self):
        pass

    def _init_optionsarea(self):
        options = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout()
        options.setLayout(hbox)

        vbox = self._init_proc_column()
        hbox.addLayout(vbox)

        vbox = self._init_const_column()
        hbox.addLayout(vbox)

        vbox = QtWidgets.QVBoxLayout()
        hbox.addLayout(vbox)
        vbox.addStretch(1)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        vbox.addWidget(button)

        return options

    def _init_proc_column(self):
        vbox = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel('Process dark runs', self)
        vbox.addWidget(label)
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('Runs:', self)
        line.addWidget(label)
        self.run1 = QtWidgets.QLineEdit('', self)
        self.run1.setFixedWidth(48)
        self.run1.setAlignment(QtCore.Qt.AlignCenter)
        self.run1.setToolTip('High gain')
        line.addWidget(self.run1)
        self.run2 = QtWidgets.QLineEdit('', self)
        self.run2.setFixedWidth(48)
        self.run2.setAlignment(QtCore.Qt.AlignCenter)
        self.run2.setToolTip('Medium gain')
        line.addWidget(self.run2)
        self.run3 = QtWidgets.QLineEdit('', self)
        self.run3.setFixedWidth(48)
        self.run3.setAlignment(QtCore.Qt.AlignCenter)
        self.run3.setToolTip('Low gain')
        line.addWidget(self.run3)
        line.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        button = QtWidgets.QPushButton('Process', self)
        button.clicked.connect(self._proc_darks)
        line.addWidget(button)
        line.addStretch(1)

        area = QtWidgets.QScrollArea(self)
        area.setWidgetResizable(True)
        vbox.addWidget(area, stretch=1)
        self.squeue_text = QtWidgets.QTextEdit('', self)
        self.squeue_text.setReadOnly(True)
        self.squeue_text.setFontPointSize(8)
        self.squeue_text.setFontFamily('Courier')
        self.squeue_text.setFontWeight(QtGui.QFont.DemiBold)
        self.squeue_text.setTabStopWidth(22)
        self.squeue_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        area.setWidget(self.squeue_text)

        self.squeue_timer = QtCore.QTimer(self)
        self.squeue_timer.timeout.connect(self._update_squeue)
        self.squeue_timer.start(1000)

        return vbox

    def _init_const_column(self):
        vbox = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel('Generate calibration constants', self)
        vbox.addWidget(label)
        label = QtWidgets.QLabel('Processed file:', self)
        vbox.addWidget(label)
        
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.proc_file = QtWidgets.QLineEdit('', self)
        line.addWidget(self.proc_file)
        button = QtWidgets.QPushButton('Browse', self)
        button.clicked.connect(self._select_proc_file)
        line.addWidget(button)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        button = QtWidgets.QPushButton('Generate', self)
        button.clicked.connect(self._gen_constants)
        line.addWidget(button)
        line.addStretch(1)

        vbox.addStretch(1)
        return vbox

    def _proc_darks(self):
        runs = [self.run1.text(), self.run2.text(), self.run3.text()]
        self.procdk.process(runs)

    def _gen_constants(self):
        self.genct.quick_agipd_calib(self.proc_file.text())

    def _update_squeue(self):
        vbar = self.squeue_text.verticalScrollBar()
        hbar = self.squeue_text.horizontalScrollBar()
        vcurr = vbar.value()
        hcurr = hbar.value()
        bottom = (vcurr == vbar.maximum())
        right = (hcurr == hbar.maximum())

        self.squeue_text.setText(subprocess.check_output(('squeue -u ' + os.environ['USER']).split()).decode())

        if bottom:
            vbar.setValue(vbar.maximum())
        else:
            vbar.setValue(vcurr)
        if right:
            hbar.setValue(hbar.maximum())
        else:
            hbar.setValue(hcurr)

    def _select_proc_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select processed file', '', 'H5 Files (*.h5);;All Files (*)')
        if fname:
            self.proc_file.setText(fname)

    def keyPressEvent(self, event): # pylint: disable=C0103
        '''Override of default keyPress event handler'''
        key = event.key()
        mod = int(event.modifiers())

        if QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+W'):
            self.close()

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = CalibGUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
