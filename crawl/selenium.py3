from selenium.webdriver import ChromeOptions
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
import time
import logging

# 实例化对象
logging.basicConfig(format="%(asctime)s %(message)s")
logging.warning("begin to test ...")
option = ChromeOptions()
option.add_experimental_option('excludeSwitches',['enable-automation'])# 开启实验性功能
# 去除特征值
option.add_argument("--disable-blink-features=AutomationControlled")
# 实例化谷歌
driver = webdriver.Chrome(options=option)
driver.set_window_size(1440, 900)
# 修改get方法
script = '''object.defineProperty(navigator,'webdriver',{undefinedget: () => undefined})'''
#execute_cdp_cmd用来执行chrome开发这个工具命令
#driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",{"source": script})

logging.warning("wait to open login page ...")
def webdriver_closed(driver):
    return driver.execute_script("return !window.navigator.webdriver")


print(driver.execute_script("var lines= []; for(var x in window){lines.push(x)};lines.sort(); return lines.join('\\\\n')"))

#print(driver.execute_script("var lines=[];return lines.join('\\\\n')"))


WebDriverWait(driver, timeout=60).until(webdriver_closed)
time.sleep(3)

logging.warning("begin to open main page ...")
driver.get('http://cpquery.cnipa.gov.cn/')


try:

	selectyzm_text_el=WebDriverWait(driver, timeout=160).until(lambda d: d.find_element(By.ID, "selectyzm_text"))
	logging.warning("has find yzm button ...")

	WebDriverWait(driver, timeout=160).until(lambda d: d.find_element(By.ID, "selectyzm_text").text.startswith('请依次点击'))

	logging.warning('element content:' + selectyzm_text_el.text)
	ActionChains(driver).move_to_element(selectyzm_text_el).perform()

	jcaptchaimage_el  = driver.find_element(By.ID, "jcaptchaimage")

	logging.warning('visible test ...')
	
	WebDriverWait(driver, timeout=10).until(lambda d:  jcaptchaimage_el.is_displayed())
	time.sleep(2)
	if jcaptchaimage_el.is_displayed():
		logging.warning('screenshot ...')
		jcaptchaimage_el.screenshot('./image.png')
		#driver.save_screenshot('./image.png')
	else:
		logging.warning('no is_displayed ...')

	time.sleep(40)
finally:
    logging.critical("exit from .......")
    driver.quit()
