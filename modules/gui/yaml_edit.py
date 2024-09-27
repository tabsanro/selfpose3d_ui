import sys
import yaml
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTreeWidget, QTreeWidgetItem, QLineEdit, QFileDialog, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt


class YAMLEditor(QWidget):
    def __init__(self, file_path=None):
        super().__init__()
        self.initUI()
        self.yaml_data = None
        self.file_path = file_path

        if self.file_path:
            self.load_yaml()

    def initUI(self):
        self.setWindowTitle('YAML Editor')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # 트리 위젯
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Key', 'Value'])
        self.tree.itemChanged.connect(self.on_item_changed)
        layout.addWidget(self.tree)

        # 버튼 레이아웃
        button_layout = QHBoxLayout()

        load_button = QPushButton('Load YAML')
        load_button.clicked.connect(self.load_yaml)
        button_layout.addWidget(load_button)

        save_button = QPushButton('Save YAML')
        save_button.clicked.connect(self.save_yaml)
        button_layout.addWidget(save_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_yaml(self):
        if not self.file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open YAML file", "", "YAML Files (*.yaml *.yml)")
            if file_path:
                self.file_path = file_path
        print(self.file_path)
        with open(self.file_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)
        self.original_data = self.yaml_data.copy()
        self.populate_tree()

    def save_yaml(self):
        if self.file_path and self.yaml_data:
            changes = self.get_changes()
            if changes:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText("The following changes will be saved:")
                msg.setInformativeText("\n".join(changes))
                msg.setWindowTitle("Save Changes")
                msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
                retval = msg.exec_()

                if retval == QMessageBox.Save:
                    with open(self.file_path, 'w') as file:
                        yaml.dump(self.yaml_data, file, default_flow_style=False)
                    self.original_data = self.yaml_data.copy()
            else:
                QMessageBox.information(self, "No Changes", "No changes to save.")

    def populate_tree(self):
        self.tree.clear()
        self.add_items(self.yaml_data, self.tree.invisibleRootItem())

    def add_items(self, value, parent):
        if isinstance(value, dict):
            for key, val in value.items():
                item = QTreeWidgetItem(parent, [str(key), ''])
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.add_items(val, item)
        elif isinstance(value, list):
            # 리스트 요소들을 부모 아이템에 직접 추가
            for i, val in enumerate(value):
                child_item = QTreeWidgetItem(parent, [str(i), str(val)])
                child_item.setFlags(child_item.flags() | Qt.ItemIsEditable)
                self.add_items(val, child_item)
        else:
            parent.setText(1, str(value))
            parent.setFlags(parent.flags() | Qt.ItemIsEditable)

    def on_item_changed(self, item, column):
        if column == 1:  # 값 열이 변경되었을 때만 처리
            self.update_yaml_data()

    def update_yaml_data(self):
        self.yaml_data = self.get_data_from_tree(self.tree.invisibleRootItem())

    def get_data_from_tree(self, item):
        if item.childCount() == 0:
            # 원래의 데이터 타입을 유지하기 위해 원본 데이터와 비교하여 적절한 형변환 적용
            value = item.text(1)
            try:
                # 숫자형 변환 시도
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                # 불리언 변환 시도
                if value.lower() in ['true', 'false']:
                    return value.lower() == 'true'
                # 기본적으로 문자열 반환
                return value
        elif all(child.text(0).isdigit() for child in [item.child(i) for i in range(item.childCount())]):
            # 모든 자식의 키가 숫자일 경우 리스트로 판단
            return [self.get_data_from_tree(item.child(i)) for i in range(item.childCount())]
        else:  # 딕셔너리 항목
            return {item.child(i).text(0): self.get_data_from_tree(item.child(i)) for i in range(item.childCount())}

    def get_changes(self):
        changes = []
        self.compare_data(self.original_data, self.yaml_data, [], changes)
        return changes

    def compare_data(self, original, current, path, changes):
        if isinstance(original, dict) and isinstance(current, dict):
            for key in set(original.keys()) | set(current.keys()):
                new_path = path + [key]
                if key not in original:
                    changes.append(f"Added: {' -> '.join(map(str, new_path))}")
                elif key not in current:
                    changes.append(f"Removed: {' -> '.join(map(str, new_path))}")
                else:
                    self.compare_data(original[key], current[key], new_path, changes)
        elif isinstance(original, list) and isinstance(current, list):
            for i in range(max(len(original), len(current))):
                new_path = path + [i]
                if i >= len(original):
                    changes.append(f"Added: {' -> '.join(map(str, new_path))}")
                elif i >= len(current):
                    changes.append(f"Removed: {' -> '.join(map(str, new_path))}")
                else:
                    self.compare_data(original[i], current[i], new_path, changes)
        elif str(original) != str(current):  # 형변환된 문자열 비교가 아닌 원래 데이터 비교로 수정
            changes.append(f"Changed: {' -> '.join(map(str, path))} from {original} to {current}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = YAMLEditor()
    editor.show()
    sys.exit(app.exec_())

