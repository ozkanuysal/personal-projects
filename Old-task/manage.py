import logging
import time
from threading import Thread

logging.basicConfig(
    filename="public/logs",
    filemode="a",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger("Dicom")


class DicomManager:
    def __init__(self, ae, target_ip, target_port, target_ae_title, roles):
        self.ae = ae
        self.target_ip = target_ip
        self.target_port = target_port
        self.target_ae_title = target_ae_title
        self.roles = roles
        self.routine_get = None

    def create_get_routine(self):
        self.routine_get = Thread(target=self.get_routine)
        self.routine_get.start()

    def get_routine(self):
        LOGGER.info("DICOM-Server: >> C-GET Routine started...")
        while True:
            time.sleep(15)
            # Add your routine logic here

    def associate(self):
        return self.ae.associate(
            self.target_ip,
            self.target_port,
            ae_title=self.target_ae_title,
            ext_neg=self.roles,
            evt_handlers=[(eval.EVT_C_STORE, self.on_c_store)],
        )
