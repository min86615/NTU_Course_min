from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class InferenceScoreWidget(QtWidgets.QDoubleSpinBox):
    def __init__(self, value=0.3):
        super(InferenceScoreWidget, self).__init__()
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setRange(0.0, 1.0)
        self.setValue(value)
        self.setToolTip("Model inference threshold(For Auto label Function)")
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignCenter)

    def minimumSizeHint(self):
        height = super(InferenceScoreWidget, self).minimumSizeHint().height()
        fm = QtGui.QFontMetrics(self.font())
        width = fm.width(str(self.maximum()))
        return QtCore.QSize(width, height)
