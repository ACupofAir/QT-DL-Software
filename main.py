from PyQt5.QtWidgets import QApplication
from colasoftware import ColaSoftware
import sys
from PyQt5.QtCore import Qt
import multiprocessing
import constants

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pool = multiprocessing.pool.ThreadPool(constants.generate_ds_data_pool_size)

    chat_window = ColaSoftware(pool)
    # chat_window.setGeometry(0, 0, 0, 0)
    chat_window.setWindowFlags(Qt.FramelessWindowHint)

    chat_window.show()
    sys.exit(app.exec_())
