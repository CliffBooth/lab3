from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, \
    QFileDialog, QTabWidget, QLineEdit, QSpinBox, QDoubleSpinBox
import sys

from model import load_model, generate_caption
from train import train

"""
we need button "load image"
 when image is loaded, it displays in the middle of the screen
 and below there is a caption

tabs on the left: train, use, settings.
in settings you can provide path for the weights and load them
"""

def testing():
    print("hello :)")

test_image = 'flickr8k/images/3637013_c675de7705.jpg'

class Worker(QObject):
    finished = pyqtSignal()
    # progress = pyqtSignal(int)

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        self.func()
        self.finished.emit()

class ImageDisplayApp(QMainWindow):
    default_weights = "pretrained_weights.h5"
    current_weights = default_weights
    model = load_model(current_weights)

    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 400, 300)
        self.setWindowTitle("Image captioning")

        ### tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { font: 14px; }")
        use_tab = QWidget()
        train_tab = QWidget()
        options_tab = QWidget()
        tabs.addTab(use_tab, "use")
        tabs.addTab(options_tab, "options")
        tabs.addTab(train_tab, "train")
        ###

        ### use tab
        use_layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.image_label.setStyleSheet("background: red;")
        use_layout.addWidget(self.image_label)

        self.caption_label = QLabel("", self)
        use_layout.addWidget(self.caption_label)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label.setStyleSheet("font: bold 14px; border: 1px solid black; max-height: 100px;")

        choose_button = QPushButton("Choose Image")
        use_layout.addWidget(choose_button)
        choose_button.clicked.connect(self.choose_image)
        use_tab.setLayout(use_layout)
        ###

        # ### train tab
        # train_layout = QVBoxLayout()
        # label = QLabel("hello, what is up my dog", self)
        # train_layout.addWidget(label)
        # train_tab.setLayout(train_layout)
        # ###

        ### options tab
        options_layout = QVBoxLayout()
        choose_file_layout = QHBoxLayout()
        self.weights_input = QLineEdit()
        choose_file_button = QPushButton("choose file")
        choose_file_button.clicked.connect(self.choose_weights)
        choose_file_button.setStyleSheet("font: 14px")
        choose_file_layout.addWidget(self.weights_input)
        choose_file_layout.addWidget(choose_file_button)
        choose_file_widget = QWidget()
        choose_file_widget.setLayout(choose_file_layout)
        choose_file_widget.setStyleSheet("max-height: auto; font: 14px")

        title_label = QLabel("choose weights file:")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("max-height: auto; font: 14px")

        load_button = QPushButton("load weights")
        def fun():
            t = self.weights_input.text()
            print(f"path: {t}")
            self.load_weights(t)

        load_button.clicked.connect(fun)
        load_button.setStyleSheet("font: 14px")

        options_layout.addWidget(title_label)
        options_layout.addWidget(choose_file_widget)
        options_layout.addWidget(load_button)
        options_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        options_tab.setLayout(options_layout)
        ###

        ### train tab
        train_layout = QVBoxLayout()
        train_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        #epochs
        epochs = QWidget()
        epochs_layout = QHBoxLayout()
        epochs.setLayout(epochs_layout)
        epochs_layout.addWidget(QLabel("epochs: "))
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setMinimum(1)
        self.epochs_spinbox.setMaximum(2 ** 31 - 1)
        self.epochs_spinbox.setValue(10)
        epochs_layout.addWidget(self.epochs_spinbox)
        train_layout.addWidget(epochs)

        #batch
        batch = QWidget()
        batch_layout = QHBoxLayout()
        batch.setLayout(batch_layout)
        batch_layout.addWidget(QLabel("batch: "))
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setMinimum(1)
        self.batch_spinbox.setMaximum(2 ** 31 - 1)
        self.batch_spinbox.setValue(32)
        batch_layout.addWidget(self.batch_spinbox)
        train_layout.addWidget(batch)

        # lr
        lr = QWidget()
        lr_layout = QHBoxLayout()
        lr.setLayout(lr_layout)
        lr_layout.addWidget(QLabel("lr: "))
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setDecimals(5)
        self.lr_spinbox.setSingleStep(0.0001)
        self.lr_spinbox.setMinimum(.0000001)
        self.lr_spinbox.setMaximum(2 ** 31 - 1)
        self.lr_spinbox.setValue(1e-4)
        lr_layout.addWidget(self.lr_spinbox)
        train_layout.addWidget(lr)

        #weights_path
        def save_file():
            file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "*.h5")
            self.save_weights_input.setText(file_path)
            pass
        weights = QWidget()
        weights_layout = QHBoxLayout()
        weights.setLayout(weights_layout)
        weights_layout.addWidget(QLabel("weights path"))
        self.save_weights_input = QLineEdit()
        button = QPushButton("choose file to save")
        button.clicked.connect(save_file)
        weights_layout.addWidget(self.save_weights_input)
        weights_layout.addWidget(button)
        train_layout.addWidget(weights)

        self.train_button = QPushButton("start training")
        self.train_button.clicked.connect(self.train)
        train_layout.addWidget(self.train_button)

        self.train_label = QLabel()
        train_layout.addWidget(self.train_label)

        train_tab.setLayout(train_layout)
        ###

        # self.setLayout(layout)
        self.setCentralWidget(tabs)

        self.set_image(test_image)

    def choose_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.gif)")
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            image_path = file_dialog.selectedFiles()[0]
            self.set_image(image_path)

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)

        pred_caption = generate_caption(image_path, self.model)
        print(f"caption: {pred_caption} | {image_path}")
        self.caption_label.setText(pred_caption)

    def choose_weights(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            path = file_dialog.selectedFiles()[0]
            self.set_weights(path)

    def set_weights(self, path):
        self.weights_input.setText(path)

    def load_weights(self, path):
        print("loading...")
        try:
            self.model = load_model(path)
        except Exception as e:
            print("error")
            print(e)
            return
        print("model loaded!")

    def train(self):
        epochs = int(self.epochs_spinbox.text())
        batch_size = int(self.batch_spinbox.text())
        path = self.save_weights_input.text()
        lr = float(self.lr_spinbox.text().replace(",", "."))
        if not path or path.isspace():
            print("empty weights path!")
            return
        print("starting training...")
        print(f"epochs: {epochs}, batch: {batch_size}, path: {path}")

        self.thread = QThread()
        def r():
            train(weights_path=path, epochs=epochs, batch_size=batch_size, learning_rate=lr)
        self.worker = Worker(r)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.on_finish_training)
        self.thread.start()

        self.train_label.setText("training...")
        self.train_button.setEnabled(False)
        pass

    def on_finish_training(self):
        print("training finished")
        self.train_label.setText("training finished")
        self.train_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = ImageDisplayApp()
    window.show()
    app.exec()


def test():
    # app = QGuiApplication(sys.argv)
    app = QApplication(sys.argv)

    image_path = "flickr8k/images/3637013_c675de7705.jpg"
    pixmap = QPixmap(image_path)

    label = QLabel()
    label.setPixmap(pixmap)
    label.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
    # test()