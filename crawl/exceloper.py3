import logging
import time

import requests
from lxml import etree
import xlwt
import csv
import xlrd
from xlutils.copy import copy

# 实例化对象

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logging.info("begin to test ...")

def crawlPage(url, params):
   result = []
   response    = requests.get('https://pro.gdstc.gd.gov.cn/egrantweb/reg-organization/toOrgSameName', params=params, timeout=10)

   if response.status_code != 200:
   	logging.error("crawl " + params + " in failure!")
   	return result
   
   root        = etree.HTML(response.content)
   contentdiv  = root.xpath("//body/div[@class='new_cont']")
   contenttext = contentdiv[0].xpath('string(.)')
   contentlist = contenttext.replace(' ','').replace('\r\n\r\n\r\n','\r\n').split('\r\n')
   
   itemindex   = -1
   for item in contentlist:
   	 itemindex +=1
   	 if item.startswith("姓名"):
   	 	logging.debug('find anchor item：' + item)
   	 	break
   	 else:
   	    logging.debug('common analysed item：' + item)
   else:
   	 logging.critical("the response format is changed!")

  
   if itemindex != -1:
   	result.append(contentlist[itemindex + 1 ])
   	result.append(contentlist[itemindex + 3 ])
   	result.append(contentlist[itemindex + 5 ])

   return result

def writeQueryResult(new_worksheet, result):
   if len(result) != 0:
   	  logging.info("姓名：" + result[0] + ",联系电话：" + result[1] + ",电子邮件：" + result[2])
   else:
   	  logging.error("query failed  with no result,please check!\r\n\r\n")
   	  return 

   write_name_cell_pos = 2
   new_worksheet.write(row, write_name_cell_pos,      result[0])
   new_worksheet.write(row, write_name_cell_pos + 1 , result[1])
   new_worksheet.write(row, write_name_cell_pos + 2,  result[2])


path = "1.xls"


try:
   workbook = xlrd.open_workbook(path)  # 打开工作簿
   sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
   worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
   rows = worksheet.nrows  # 获取表格中已存在的数据的行数

   logging.info("create new sheet for write form old file ...")
   new_workbook  = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
   new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格

   logging.debug("first row is logo and second row is headers ...")

   for row in range(2, 20):
   	 orgName = worksheet.cell_value(row, 0)
   	 logging.info("cellvalue:" + orgName)
   	 result  = crawlPage('https://pro.gdstc.gd.gov.cn/egrantweb/reg-organization/toOrgSameName', {'orgName': orgName})	
   	 writeQueryResult(new_worksheet, result)

   logging.info("save query results ...")
   new_workbook.save(path)  # 保存工作簿


finally:
   logging.warning("exit from process !!!")
