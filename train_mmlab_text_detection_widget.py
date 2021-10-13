# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_mmlab_text_detection.train_mmlab_text_detection_process import TrainMmlabTextDetectionParam
from train_mmlab_text_detection.utils import textdet_models
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class TrainMmlabTextDetectionWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainMmlabTextDetectionParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model name
        self.combo_model_name = pyqtutils.append_combo(self.grid_layout, "Model name")
        for name in textdet_models.keys():
            self.combo_model_name.addItem(name)
        self.combo_model_name.setCurrentText(self.parameters.cfg["model_name"])

        # Epochs
        self.spin_epochs = pyqtutils.append_spin(self.grid_layout, "Epochs", self.parameters.cfg["epochs"])

        # Batch size
        self.spin_batch = pyqtutils.append_spin(self.grid_layout, "Batch size", self.parameters.cfg["batch_size"])

        # Evaluation period
        self.spin_eval_period = pyqtutils.append_spin(self.grid_layout, "Eval period",
                                                      self.parameters.cfg["eval_period"])

        # Ratio train test
        self.spin_train_test = pyqtutils.append_spin(self.grid_layout, "Train test percentage",
                                                     self.parameters.cfg["dataset_split_ratio"])

        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(self.grid_layout, label="Output folder",
                                                              path=self.parameters.cfg["output_folder"],
                                                              tooltip="Select folder",
                                                              mode=QFileDialog.Directory)

        # Dataset folder
        self.browse_dataset_folder = pyqtutils.append_browse_file(self.grid_layout, label="Dataset folder",
                                                                  path = self.parameters.cfg["dataset_folder"],
                                                                  tooltip="Select folder",
                                                                  mode=QFileDialog.Directory)
        # Expert mode
        self.check_expert = pyqtutils.append_check(self.grid_layout, "Expert mode", self.parameters.cfg["expert_mode"])
        self.check_expert.stateChanged.connect(self.on_expert_mode_change)

        # Custom Model
        self.label_model = QLabel("Model config file (.py)")
        self.browse_cfg_file = pyqtutils.BrowseFileWidget(path=self.parameters.cfg["custom_model"],
                                                          tooltip="Select file",
                                                          mode=QFileDialog.ExistingFile)
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_model, row, 0)
        self.grid_layout.addWidget(self.browse_cfg_file, row, 1)

        # Pretrain
        self.label_pretrain = QLabel("Pretrain weight file (.pth)")
        self.browse_pretrain_file = pyqtutils.BrowseFileWidget(path=self.parameters.cfg["pretrain"],
                                                               tooltip="Select file",
                                                               mode=QFileDialog.ExistingFile)

        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_pretrain, row, 0)
        self.grid_layout.addWidget(self.browse_pretrain_file, row, 1)

        self.label_model.setVisible(self.check_expert.isChecked())
        self.browse_cfg_file.setVisible(self.check_expert.isChecked())
        self.label_pretrain.setVisible(self.check_expert.isChecked())
        self.browse_pretrain_file.setVisible(self.check_expert.isChecked())

        self.spin_batch.setVisible(not(self.check_expert.isChecked()))
        self.spin_epochs.setVisible(not(self.check_expert.isChecked()))
        self.spin_eval_period.setVisible(not(self.check_expert.isChecked()))
        self.combo_model_name.setVisible(not(self.check_expert.isChecked()))

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def on_expert_mode_change(self, int):
        self.label_model.setVisible(self.check_expert.isChecked())
        self.browse_cfg_file.setVisible(self.check_expert.isChecked())
        self.label_pretrain.setVisible(self.check_expert.isChecked())
        self.browse_pretrain_file.setVisible(self.check_expert.isChecked())

        self.spin_batch.setVisible(not(self.check_expert.isChecked()))
        self.spin_epochs.setVisible(not(self.check_expert.isChecked()))
        self.spin_eval_period.setVisible(not(self.check_expert.isChecked()))
        self.combo_model_name.setVisible(not(self.check_expert.isChecked()))

    def onApply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.cfg["model_name"] = self.combo_model_name.currentText()
        self.parameters.cfg["epochs"] = self.spin_epochs.value()
        self.parameters.cfg["batch_size"] = self.spin_batch.value()
        self.parameters.cfg["eval_period"] = self.spin_eval_period.value()
        self.parameters.cfg["dataset_split_ratio"] = self.spin_train_test.value()
        self.parameters.cfg["expert_mode"] = self.check_expert.isChecked()
        self.parameters.cfg["custom_model"] = self.browse_cfg_file.path
        self.parameters.cfg["dataset_folder"] = self.browse_dataset_folder.path
        self.parameters.cfg["output_folder"] = self.browse_out_folder.path
        self.parameters.cfg["pretrain"] = self.browse_pretrain_file.path

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainMmlabTextDetectionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_mmlab_text_detection"

    def create(self, param):
        # Create widget object
        return TrainMmlabTextDetectionWidget(param, None)