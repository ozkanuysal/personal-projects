import logging


class AppLogger:
    """
    A class to create and configure a logger for an application.

    Attributes
    ----------
    logger : logging.Logger
        The logger object used for logging.

    Methods
    -------
    __init__(name: str)
        Initializes the logger with the given name, sets the log level to DEBUG,
        and adds a StreamHandler with a specific formatter.

    get_logger() -> logging.Logger
        Returns the configured logger object.
    """

    def __init__(self, name: str):
        """
        Initializes the logger with the given name, sets the log level to DEBUG,
        and adds a StreamHandler with a specific formatter.

        Parameters
        ----------
        name : str
            The name of the logger.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger object.

        Returns
        -------
        logging.Logger
            The configured logger object.
        """
        return self.logger