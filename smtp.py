#!/usr/bin/env python
# -*- coding:utf-8 -*-
import smtplib
import os  
import sys
import datetime  
import time  
import random
import glob  
import shutil  
import string  
import os.path  
import openpyxl
import shutil
from openpyxl import load_workbook
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.MIMEBase import MIMEBase
from email.header import Header
from email import Utils, Encoders



#datetime calc
now         =  datetime.datetime.now()
curmonth    = int(now.strftime('%m'))  
salarymonth = curmonth - 1

#excel process
curdir         = os.getcwd().decode('utf-8')
print('curdir:' + curdir)
salarymonthdir = curdir + os.sep  + "salarys" + os.sep + str(curmonth)

#clear files
if os.path.exists(salarymonthdir):
 shutil.rmtree(salarymonthdir)
os.makedirs(salarymonthdir)
salarytemplate_filename = 'template.xlsx'

#find the salary_sum file
xlsxfile_count = 0
xlsx_files     = os.listdir(curdir)
for xlsxfile in xlsx_files:
  if os.path.isfile(xlsxfile) and xlsxfile.find('.xlsx') != -1:
    #print('isFile:' + xlsxfile)
    if xlsxfile.find(salarytemplate_filename) == -1:
      salary_sum_file = xlsxfile
    xlsxfile_count = 1 + xlsxfile_count



if xlsxfile_count > 2:
  print("*.xlsx(s) in the directory exceed two,can't recognize the salary_sum_file!!")
  print('''\
        ****************usage*************************
        parentdir
        |--any xlsx file(exclude template.xlsx,use as the salary_sum_file)
        |--template.xlsx
        |--smtp.py\
        ''')
  exit(0)

print("meet the request conditions(*.xlsx files:%d),to run...................."%xlsxfile_count)


filename                = salary_sum_file


#data_only: controls whether cells with formulae have either the formula (default) 
#or the value stored the last time Excel read the sheet   
mybook        = load_workbook(filename,read_only=True,data_only=True)
sheetnames    = mybook.get_sheet_names()
lastsheetname = sheetnames[-1]
#default salary month is the last one
salary_sheet_sum      = mybook[lastsheetname]
salary_sheet_sum_cell = salary_sheet_sum.cell


#mail process
charset    = 'gb2312'
receiver   = 'changyunleitest@126.com'
sender     = 'ai.dangmei@astute-tec.com'
smtpserver = 'smtp.mxhichina.com'

#sender     = 'njaixin@163.com'
#smtpserver = 'smtp.163.com'
username   = sender
password   = 'njaixin@china.com.cn'

send_mail_count = 0


name_col_pos  = 2
email_col_pos = 22;
#you should change the value src col --> dst col 
copyCells = [
            (2,1),
            (4,2),
            (5,3),
            (6,4),
            (7,5),#days
            (8,6),
            (9,7),
            (10,8),
            (11,9),
            (12,10),
            (13,11),
            (14,12),
            (15,13),
            (16,14),
            (17,15),
            (18,16),
            (20,17),#remarks
            ]

#salary_sheet_sum.max_row 
smtp = None
for row in range(3,salary_sheet_sum.max_row):
  if send_mail_count == 0:
    smtp          = smtplib.SMTP(smtpserver,25)
    smtp.login(username, password) 
  name            = salary_sheet_sum_cell(row=row,column=name_col_pos).value
  
  if not name:
  	print('meet the first empty name cell,reach the end of salary table.......')
  	break

  subject         = u'工资明细'
  salary_filename = name + u'.xlsx' 
  #print(salary_filename)
  salary_filename_full   = (salarymonthdir) + (os.sep) 
  salary_filename_full   = salary_filename_full+salary_filename
  print(salary_filename_full)
  shutil.copy(salarytemplate_filename,salary_filename_full)
  salary_book            = load_workbook(salary_filename_full)
  salary_book_sheetnames = salary_book.get_sheet_names()
  salary_book_sheet      = salary_book[salary_book_sheetnames[len(salary_book_sheetnames)-1]]
  #copy cell
  for copycelcfg in copyCells:
    value = salary_sheet_sum_cell(row=row,column=copycelcfg[0]).value 
  	#print(u'copy value:' + unicode(value))
    if not value:
      continue
    salary_book_sheet.cell(row=2,column=copycelcfg[1]).value = value
  #save everyone's salary book  
  salary_book.save(salary_filename_full)  	
  salary_book.close()  
  
  #send email has attached file,must be mixed
  msgRoot            = MIMEMultipart('mixed')
  msgRoot['Subject'] = subject
  msgRoot['From']    = sender   #must field,maybe regard as spam

  email = salary_sheet_sum_cell(row=row,column=email_col_pos).value
  if not email :
    email = receiver
  msgRoot['To']      = email

  #content
  msText = MIMEText(u'您的工资明细•‿•',_subtype='plain',_charset='utf-8')
  msgRoot.attach(msText)

  #attachment
  maintype = 'application'
  subtype  = 'octet-stream'
  att      = MIMEBase(maintype,subtype)
  att.set_payload(open(salary_filename_full, 'rb').read())
  Encoders.encode_base64(att)
 
  
  att["Content-Disposition"] = 'attachment; filename= %s'%Header(u'工资明细.xlsx','UTF-8')
  #print('attachment:' + att["Content-Disposition"])
  msgRoot.attach(att)
  
 
  print('send mail to:' + email + ',from:' + sender)  
  smtp.sendmail(sender, email, msgRoot.as_string())
  #avoid to be recoginized as spam
  time.sleep(random.randint(1,3))
  send_mail_count = send_mail_count + 1

  #avoid sending too much mails on the same login session will to be recoginized as spam
  if send_mail_count >= 9:
    smtp.quit()
    send_mail_count = 0
    smtp = None
    print('sleep to next send mail loop')
    time.sleep(random.randint(12,20))

#finish operation
if  smtp :
  smtp.quit()
mybook.close()








