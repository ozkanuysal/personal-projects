import pypyodbc as db
import json

class DBClass:
    def __init__(self):
        self.driver = ''
        self.trusted_connection = ''
        self.server = ''
        self.database = ''
        self.username = ''
        self.password = ''
        self.connection = None
        self.cursor = None

    def get_open_cursor(self, json_string):
        self.prepare_connection_detail(json_string)
        self.prepare_connection()
        return self.cursor

    def close_connection(self):
        if self.connection:
            self.connection.close()
            print('Connection closed.')

    def prepare_connection_detail(self, dbconfig_json):
        db_info = dbconfig_json['dbInfo']
        self.driver = db_info.get('driver', '')
        self.server = db_info.get('server', '')
        self.database = db_info.get('database', '')
        self.username = db_info.get('username', '')
        self.password = db_info.get('password', '')
        self.trusted_connection = db_info.get('trusted_connection', '')
    
    def prepare_connection(self):
        connection_string = (
            f'DRIVER={self.driver};'
            f'SERVER={self.server};'
            f'DATABASE={self.database};'
            f'UID={self.username};'
            f'PWD={self.password};'
        )
        self.connection = db.connect(connection_string)   
        self.cursor = self.connection.cursor()