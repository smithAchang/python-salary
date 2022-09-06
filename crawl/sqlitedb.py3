# -*- coding: utf-8 -*-


import time
from datetime import datetime
from datetime import timedelta
import logging
import sqlite3
from sqlite3 import Error

import unittest


class MyDB:
    def __init__(self,):
        self.con    = sqlite3.connect('mydatabase.db')
        self.cursor = self.con.cursor()

    def createTables(self):
        try:

            self.cursor.execute('''
                                 create table if not exists "company_query_status"(
                                      "name" TEXT NOT NULL UNIQUE,
                                      "status" TEXT NOT NULL,
                                      "date" DATE NOT NULL
                                    )

                               ''')

            self.cursor.execute('''
                                    create table if not exists "reject_patent"(
                                      "no" TEXT NOT NULL UNIQUE,
                                      "name" TEXT NOT NULL ,
                                      "company_name" TEXT NOT NULL,
                                      "reject_date" DATE NOT NULL
                                    )

                               ''')

            self.con.commit()

        except Error as e:
            logging.error(e)

    def closeDB(self):
        try:
            self.con.commit()
            self.con.close()

            self.cursor = None
            self.con    = None

        except Error as e:
            logging.error(e)

    def dropTables(self):
        try:
            self.cursor.execute("drop table if exists company_query_status")
            self.cursor.execute("drop table if exists reject_patent")
            self.con.commit()

        except Error as e:
            logging.error(e)

    def modCompanyQuueryStatus(self, company_name):
        self.cursor.execute("select count(*) from company_query_status where name='%s'"%company_name)
        rows   = self.cursor.fetchall()

        now    = datetime.now()
        nowStr = datetime.strftime(now, "%Y-%m-%d")

        if len(rows) == 0 :
           self.cursor.execute("insert into company_query_status values('%s','%s','%s')"%(company_name, status, nowStr))    
        else:
           self.cursor.execute("update company_query_status set status = '%s', date = '%s' where name='%s'"%(status, nowStr))    
        
        self.con.commit()

    def isCompanyHasBeenQueryed(self, company_name):
        self.cursor.execute("select status from company_query_status where name='%s'"%company_name)
        rows = self.cursor.fetchall()

        if len(rows) == 0 :
            return False

        status = rows[0]

        if status != "finished":
            return False

        return True

    def deleteOldData(self):
        now    = datetime.now()
        nowStr = datetime.strftime(now, "%Y-%m-%d")
        self.cursor.execute("delete from company_query_status where date < '%s'"%nowStr)

        threemonthago    = now - timedelta(days=90)
        threemonthagoStr = datetime.strftime(threemonthago, "%Y-%m-%d")
        self.cursor.execute("delete from reject_patent where date < '%s'"%threemonthagoStr)
             
'''
  Test Case
'''

class TestMyDB(unittest.TestCase):

    def setUp(self):
        logging.info('setUp...')
        self.db = MyDB()
        self.db.dropTables()
        self.db.createTables()

    def tearDown(self):
        logging.info('tearDown...')
        self.db.closeDB()
        self.db = None


    def test_HasBeenQueryed(self):
        
        self.assertEqual(self.db.isCompanyHasBeenQueryed("abc"), False)
        

   



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("test case begin to run ...")
    unittest.main()

    


           
