import pypyodbc as db
import json


class DBClass(object):
    driver = ''
    trusted_connection = ''
    server = ''
    database = ''
    username = ''
    password = ''

    def __init__(self):
       pass

    def GetOpenCurser(self, json_string):
        self.PrepareConnectionDetail(json_string)
        self.PrepareConnection()
        return self.cursor

    def CloseConnection(self):
        self.connection.close()
        print('closed ')

    def PrepareConnectionDetail(self, dbconfigJson):
        print(dbconfigJson)
        self.driver= dbconfigJson['dbInfo']['driver']
        self.server= dbconfigJson['dbInfo']['server']
        self.database= dbconfigJson['dbInfo']['database']
        self.username= dbconfigJson['dbInfo']['username']
        self.password= dbconfigJson['dbInfo']['password']
        self.trusted_connection= dbconfigJson['dbInfo']['trusted_connection']
    
    def PrepareConnection(self):
        self.connectionstring = f'DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password};'
        print(self.connectionstring)
        self.connection = db.connect(self.connectionstring)   
        self.cursor = self.connection.cursor()


        
