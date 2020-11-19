"""
This section contains the code necessary to run the GUI relating to the Option Price Calculator. The aim is to
receive user input for the option and underlying stock details, then calling the functions in 'option_calc.py'
to solve for options price and Greeks.
"""

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
from option_calc import EuropeanOption

qtCreatorFile = "OptionPriceCalculatorGUI.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.comboBox_method.addItems(['Explicit', 'Implicit', 'Crank-Nicolson'])
        self.pushButton_calculate.clicked.connect(self.calculate)

    def calculate(self):
        S = float(self.lineEdit_S.text())
        K = float(self.lineEdit_K.text())
        T = self.dateEdit_Value.date().daysTo(self.dateEdit_Expiration.date())
        r = float(self.lineEdit_r.text())
        q = float(self.lineEdit_q.text())
        N = int(self.lineEdit_N.text())
        M = int(self.lineEdit_M.text())
        sigma = float(self.lineEdit_sigma.text())

        option = EuropeanOption(S, K, T, r, q, sigma)
        if self.comboBox_method.currentText() == 'Explicit':
            value = option.explicit_method(N, M)
        elif self.comboBox_method.currentText() == 'Implicit':
            value = option.implicit_method(N, M)
        elif self.comboBox_method.currentText() == 'Crank-Nicolson':
            value = option.crank_n_method(N, M)
        greeks = option.calc_greeks()

        # Call and Put Option Price output
        self.label_call.setText(str(round(value['call'], 3)))
        self.label_put.setText(str(round(value['put'], 3)))

        # Greeks output
        self.label_call_d.setText(str(round(greeks["delta_c"], 3)))
        self.label_call_g.setText(str(round(greeks["gamma_c"], 3)))
        self.label_call_v.setText(str(round(greeks["vega_c"], 3)))
        self.label_call_t.setText(str(round(greeks["theta_c"], 3)))
        self.label_call_r.setText(str(round(greeks["rho_c"], 3)))
        self.label_put_d.setText(str(round(greeks["delta_p"], 3)))
        self.label_put_g.setText(str(round(greeks["gamma_p"], 3)))
        self.label_put_v.setText(str(round(greeks["vega_p"], 3)))
        self.label_put_t.setText(str(round(greeks["theta_p"], 3)))
        self.label_put_r.setText(str(round(greeks["rho_p"], 3)))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
