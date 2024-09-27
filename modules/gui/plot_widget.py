# # TODO 1. 실시간으로 들어오는 3D 인물 포즈를 표시하는 기능 (비동기 처리)
# # TODO 2. 각종 정보들을 표시하기 위한 그룹박스를 추가
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

DISTANCE = 100000

LIMBS = [
    [0, 1],
    [0, 2],
    [0, 3],
    [3, 4],
    [4, 5],
    [0, 9],
    [9, 10],
    [10, 11],
    [2, 6],
    [2, 12],
    [6, 7],
    [7, 8],
    [12, 13],
    [13, 14],
]

class PoseUpdater(QtCore.QThread):
    pose_updated = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.distance = 0

    def update_pose(self, results=None):
        self.pose_updated.emit(results)
        self.msleep(1)
    
    def set_distance_value(self, value):
        """Method to update slider value in PoseUpdater thread"""
        self.distance = value

class PlotWidget(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.setWindowTitle('3D Pose Estimation with Overwriting Signals')

        # Main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        # Left side: Group box for information display
        self.info_groupbox = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout()

        # Add label to show pose information
        self.info_label = QtWidgets.QLabel("Pose info will be displayed here.")
        info_layout.addWidget(self.info_label)

        # Add slider for variable adjustment
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(3000)
        self.slider.setValue(0)  # Default value
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        info_layout.addWidget(self.slider)

        # Add label to show slider value
        self.slider_value_label = QtWidgets.QLabel("LoD distance(mm) : 0")
        info_layout.addWidget(self.slider_value_label)

        self.info_groupbox.setLayout(info_layout)

        # Right side: 3D plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')

        # Layouts
        layout.addWidget(self.info_groupbox)
        layout.addWidget(self.canvas)

        # Pose Updater (for real-time pose updates)
        self.pose_updater = PoseUpdater()
        self.pose_updater.pose_updated.connect(self.update_pose)
        self.pose_updater.start()

        # Flag to control whether we are processing or not
        self.processing = False
        self.pending_data = None

    def draw_cylinder(self, ax, center, height=200, radius=50, resolution=20):
        """Draw a cylinder at the given center (x, y, z)."""
        theta = np.linspace(0, 2 * np.pi, resolution)
        z = np.linspace(0, center[2] + height, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)

        x_grid = center[0] + radius * np.cos(theta_grid)
        y_grid = center[1] + radius * np.sin(theta_grid)

        # Draw the cylindrical surface
        ax.plot_surface(x_grid, y_grid, z_grid, color='b', alpha=0.5, rstride=4, cstride=4)


    def update_pose(self, results):
        if self.processing:
            # Store the most recent data and return
            self.pending_data = (results)
            return

        # Mark as processing
        self.processing = True

        if not results:
            # No pose data available
            return

        # Clear the previous plot
        self.ax.clear()

        for single_pose in results:
            if single_pose['root'][3] == -1:
                continue
            if single_pose['root'][3] == 1:
                x = single_pose['root'][0]
                y = single_pose['root'][1]
                z = single_pose['root'][2]

                self.draw_cylinder(self.ax, [x, y, z], height=700, radius=100) 

            elif single_pose['root'][3] == 0:
                x = single_pose['pred'][:, 0]
                y = single_pose['pred'][:, 1]
                z = single_pose['pred'][:, 2]

                self.ax.scatter(x, y, z, c='r', marker='o')

                for limb in LIMBS:
                    joint1, joint2 = limb
                    x_values = [single_pose['pred'][joint1][0], single_pose['pred'][joint2][0]]
                    y_values = [single_pose['pred'][joint1][1], single_pose['pred'][joint2][1]]
                    z_values = [single_pose['pred'][joint1][2], single_pose['pred'][joint2][2]]
                    self.ax.plot(x_values, y_values, z_values, 'ro-')

        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        self.ax.set_xlim(-2500, 2500)  # x축 범위 설정
        self.ax.set_ylim(-2500, 2500)  # y축 범위 설정
        self.ax.set_zlim(0, 2000)  # z축 범위 설정

        # Redraw the canvas
        self.canvas.draw()

        # Mark as done processing and check if new data arrived
        self.processing = False
        if self.pending_data:
            # Process the most recent data immediately
            latest_results = self.pending_data
            self.pending_data = None
            self.update_pose(latest_results)

    def on_slider_value_changed(self):
        # Get the current slider value and update the label
        value = self.slider.value()
        self.slider_value_label.setText(f"LoD distance(mm) : {value}")
        
        self.pose_updater.set_distance_value(value)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = PlotWidget()
    mainWindow.show()
    sys.exit(app.exec_())
