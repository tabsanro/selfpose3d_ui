import sys
import os.path as osp
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore
from datetime import datetime

from modules.gui.yaml_edit import YAMLEditor
from modules.gui.camera_widget import CameraWidget
from modules.gui.record_widget import RecordWidget

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치
# form_class = uic.loadUiType(osp.join(osp.dirname(__file__),"pushbutton.ui"))[0]

base_dir = osp.dirname(osp.abspath(__file__))
form_class = uic.loadUiType(osp.join(base_dir, 'modules', 'gui', 'poc_ui.ui'))[0]

today = datetime.today()
formated_today = today.strftime("%m%d")

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Source_Folder.hide()
        self.WebCam_Source.hide()
        self.Mode_Record.hide()

        self.radioButton.clicked.connect(self.groupdboxRadFunction)
        self.radioButton_2.clicked.connect(self.groupdboxRadFunction)
        self.radioButton_3.clicked.connect(self.groupdboxRadFunction)
        self.radioButton_4.clicked.connect(self.groupdboxRadFunction)

        self.folder_button.clicked.connect(self.open_folder)
        self.folder_button_2.clicked.connect(self.open_folder2)

        self.config_edit_button.clicked.connect(self.open_yaml_edit)

        self.run_button.clicked.connect(self.run)
        self.streamButton.clicked.connect(self.stream)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)

        self.folder_path.setText(osp.join(base_dir, "modules", "SelfPose3d", "data_0705"))
        self.source_path.setText(osp.join(base_dir, "modules", "realsense", "realsense_info.json"))
        self.record_path.setText(osp.join(base_dir, "modules", "SelfPose3d", f"data_{formated_today}"))
        self.config_path.setText(osp.join(base_dir, "configs", "focus.yaml"))

    def open_yaml_edit(self):
        self.editor = YAMLEditor(self.config_path.text())
        self.editor.show()

    def groupdboxRadFunction(self):
        if self.radioButton.isChecked():
            self.WebCam_Source.hide()
            self.Source_Folder.show()
            self.Mode_Record.hide()
            self.config_box.show()
        elif self.radioButton_2.isChecked(): 
            self.Source_Folder.hide()
            self.WebCam_Source.show()
            self.Mode_Record.hide()
            self.config_box.show()
        elif self.radioButton_3.isChecked(): self.Source_Folder.show()
        elif self.radioButton_4.isChecked():
            self.WebCam_Source.show()
            self.Source_Folder.hide()
            self.Mode_Record.show()
            self.config_box.hide()

    def open_folder(self):
        # 폴더 선택 대화 상자 열기
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path.setText(folder)
    
    def open_folder2(self):
        # 폴더 선택 대화 상자 열기
        file, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file:
            self.config_path.setText(file)
        
    def stream(self):
        print("stream will be shown")
        self.editor = CameraWidget(self.source_path.text())
        self.editor.show()

    def run(self):
        print("Run button clicked")
        
        if self.radioButton.isChecked():
            cmd_args = [
                "--cfg_focus", self.config_path.text(),
                "--cfg_sp3d", "modules/SelfPose3d/config/cam4_posenet.yaml",
                "--source_folder", self.folder_path.text()
            ]
        elif self.radioButton_2.isChecked():
            cmd_args = [
                "--cfg_focus", self.config_path.text(),
                "--cfg_sp3d", "modules/SelfPose3d/config/cam4_posenet.yaml",
                "--webcam=True"
                "--webcam_info", self.source_path.text()
            ]
        elif self.radioButton_4.isChecked():
            print("Record mode will be started")
            self.editor = RecordWidget(cam_info_path=self.source_path.text(), path=self.record_path.text())
            self.editor.show()
            return
        else:
            print("Please select a mode")
            return

        script_path = osp.join(base_dir, "run.py")
        print("python", [script_path] + cmd_args)
        self.process.start("python", [script_path] + cmd_args)

    def handle_stdout(self):
        # 표준 출력 읽기
        output = self.process.readAllStandardOutput().data().decode()
        self.logTextEdit.append(output)

    def handle_stderr(self):
        # 표준 에러 읽기
        error = self.process.readAllStandardError().data().decode()
        print(f"Raw error: '{error}'") 
        if error.strip():
            self.logTextEdit.append("Error: " + error)

if __name__ == "__main__":
    #QApllication: 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass()
    #프로그램 화면을 보여주는 코드
    myWindow.show()
    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()