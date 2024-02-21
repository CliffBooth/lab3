from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, \
    QFileDialog, QTabWidget
import sys

from model import load_model, generate_caption

"""
we need button "load image"
 when image is loaded, it displays in the middle of the screen
 and below there is a caption

tabs on the left: train, use, settings.
in settings you can provide path for the weights and load them
"""

def on_load_image():
    print("hello :)")

test_image = 'flickr8k/images/3637013_c675de7705.jpg'



class ImageDisplayApp(QMainWindow):
    default_weights = "pretrained_weights.h5"
    model = load_model(default_weights)

    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 400, 300)
        self.setWindowTitle("Image captioning")

        ### tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { font: 14px; }")
        use_tab = QWidget()
        train_tab = QWidget()
        tabs.addTab(use_tab, "use")
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

        ### train tab
        train_layout = QVBoxLayout()
        label = QLabel("hello, what is up my dog", self)
        train_layout.addWidget(label)
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