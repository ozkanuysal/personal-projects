import time
from threading import Thread

from logger_ import create_logger
from pynetdicom import evt

LOGGER = create_logger("Dicom")

class DicomManager:
    def __init__(self, ae, target_ip, target_port, target_ae_title, roles):
        """
        Initializes the DicomManager with the specified parameters.

        Parameters
        ----------
        ae : pynetdicom.AE
            The AE to use for the DICOM server.
        target_ip : str
            The IP address of the target to associate with.
        target_port : int
            The port number of the target to associate with.
        target_ae_title : str
            The AE title of the target to associate with.
        roles : List[str]
            A list of roles for the AE to negotiate.

        Attributes
        ----------
        routine_get : Optional[Thread]
            A thread for running the get routine in the background, initialized as None.
        """

        self.ae = ae
        self.target_ip = target_ip
        self.target_port = target_port
        self.target_ae_title = target_ae_title
        self.roles = roles
        self.routine_get = None

    def create_get_routine(self):
        """
        Creates and starts a separate thread to run the get routine in the background.

        This method initializes a new thread with the target set to the `get_routine`
        method and starts it in daemon mode, allowing it to run in the background
        while the main program continues execution.

        Returns
        -------
        None
        """

        self.routine_get = Thread(target=self.get_routine, daemon=True)
        self.routine_get.start()

    def associate(self):
        """
        Associate with the target AE.

        Parameters
        ----------
        None

        Returns
        -------
        assoc : pynetdicom.association.Association
            The Association object representing the DICOM association.
        """

        return self.ae.associate(
            self.target_ip,
            self.target_port,
            ae_title=self.target_ae_title,
            ext_neg=self.roles,
            evt_handlers=[(evt.EVT_C_STORE, self.on_c_store)],
        )

    def start(self):
        """
        Starts the DICOM server. This method will start the get routine, the server
        and associate with the target AE. The server will run in non-blocking mode.

        Returns
        -------
        None
        """

        self.create_get_routine()
        self.ae.start_server((self.target_ip, self.target_port), block=False)
        self.associate()
        LOGGER.info("DICOM-Server: >> Server started...")
    
    def stop(self):
        """
        Stops the DICOM server.

        This method will stop the server and shutdown the association with the target AE.

        Returns
        -------
        None
        """

        self.ae.shutdown()
        LOGGER.info("DICOM-Server: >> Server stopped...")

def start_dicom_server(ae, target_ip, target_port, target_ae_title, roles):
    """
    Starts a DICOM server on the given AE.

    Parameters
    ----------
    ae : pynetdicom.AE
        The AE to use for the DICOM server.
    target_ip : str
        The IP address of the target to associate with.
    target_port : int
        The port number of the target to associate with.
    target_ae_title : str
        The AE title of the target to associate with.
    roles : List[str]
        A list of roles for the AE to negotiate.

    Returns
    -------
    DicomManager
        The DicomManager instance that was created.
    """

    dicom_manager = DicomManager(ae, target_ip, target_port, target_ae_title, roles)
    dicom_manager.start()
    return dicom_manager

def stop_dicom_server(dicom_manager):
    
    """
    Stops a DICOM server started by start_dicom_server.

    Parameters
    ----------
    dicom_manager : DicomManager
        The DicomManager instance returned by start_dicom_server.

    Notes
    -----
    This function blocks until the DICOM server is stopped and the
    association is released.
    """

    dicom_manager.stop()

def start_dicom_server_thread(ae, target_ip, target_port, target_ae_title, roles):
    """
    Starts a DICOM server on the given AE in a separate thread.

    Parameters
    ----------
    ae : pynetdicom.AE
        The AE to use for the DICOM server.
    target_ip : str
        The IP address of the target to associate with.
    target_port : int
        The port number of the target to associate with.
    target_ae_title : str
        The AE title of the target to associate with.
    roles : List[str]
        A list of roles for the AE to negotiate.

    Returns
    -------
    DicomManager
        The DicomManager instance that was created.
    """

    dicom_manager = DicomManager(ae, target_ip, target_port, target_ae_title, roles)
    dicom_manager.create_get_routine()
    dicom_manager.start()
    return dicom_manager
