'''
selenium_crawler.py
创建时间：2020-08-15
创建者：Alvin

脚本功能：
    使用selenium库进行的爬虫demo代码

参考代码：https://www.cnblogs.com/wsmrzx/p/9556543.html
'''

from selenium import webdriver
import selenium
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import time


class Spider():
    def open_browser(self):
        # 若下列命令报错，请进入下面链接下载chromedriver然后放置在/user/bin/下即可
        # https://chromedriver.storage.googleapis.com/index.html?path=2.35/
        self.options = webdriver.ChromeOptions()
        self.options.add_argument(self.user_agent)
        self.browser = webdriver.Chrome(options = self.options)
        # 隐式等待：等待页面全部元素加载完成（即页面加载圆圈不再转后），才会执行下一句，如果超过设置时间则抛出异常
        try:
            self.browser.implicitly_wait(10)
        except:
            print("页面无法加载完成，无法开启爬虫操作！")
        # 显式等待：设置浏览器最长允许超时的时间
        self.wait = WebDriverWait(self.browser, 10)


    def init_variable(self, url_link, search_key, user_agent):
        # url_link为电商平台首页，search_key为商品搜索词
        self.url = url_link
        self.keyword = search_key
        self.isLastPage = False
        self.user_agent = user_agent
        print("###############\n##初始化参数##\n###############\nurl：%s\nkeyword：%s\nuser_agent：%s\n\n" % (self.url, self.keyword, self.user_agent))


    def parse_JDpage(self):
        try:
            # 定位元素并获取元素下的字段值（商品标题，价格，评论数，商品链接）
            names = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="gl-i-wrap"]/div[@class="p-name p-name-type-2"]/a/em')))
            prices = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="gl-i-wrap"]/div[@class="p-price"]/strong/i')))
            comment_nums = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="gl-i-wrap"]/div[@class="p-commit"]/strong')))
            links = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//li[@class="gl-item"]')))
            page_num = self.wait.until(EC.presence_of_element_located((By.XPATH, '//div[@class="page clearfix"]//input[@class="input-txt"]')))
            names = [item.text for item in names]
            prices = [price.text for price in prices]
            comment_nums = [comment_num.text for comment_num in comment_nums]
            links = ["https://item.jd.com/{sku}.html".format(sku = link.get_attribute("data-sku")) for link in links]
            page_num = page_num.get_attribute('value')
        except selenium.common.exceptions.TimeoutException:
            print('parse_page: TimeoutException 网页超时')
            self.parse_JDpage()
        except selenium.common.exceptions.StaleElementReferenceException:
            print('turn_page: StaleElementReferenceException 某元素因JS刷新已过时没出现在页面中')
            print('刷新并重新解析网页...')
            self.browser.refresh()
            self.parse_JDpage()
            print('解析成功')
        return names, prices, comment_nums, links, page_num


    def turn_JDpage(self):
        # 移到页面末端并点击‘下一页’
        try:
            self.browser.find_element_by_xpath('//a[@class="pn-next" and @onclick]').click()
            time.sleep(1) # 点击完等1s
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2) # 下拉后等2s
        # 如果找不到元素，说明已到最后一页
        except selenium.common.exceptions.NoSuchElementException:
            self.isLastPage = True
        # 如果页面超时，跳过此页
        except selenium.common.exceptions.TimeoutException:
            print('turn_page: TimeoutException 网页超时')
            self.turn_JDpage()
        # 如果因为JS刷新找不到元素，重新刷新
        except selenium.common.exceptions.StaleElementReferenceException:
            print('turn_page: StaleElementReferenceException 某元素因JS刷新已过时没出现在页面中')
            print('刷新并重新翻页网页...')
            self.browser.refresh()
            self.turn_JDpage()
            print('翻页成功')


    def JDcrawl(self, url_link, search_key, user_agent, save_path):
        # 初始化参数
        self.init_variable(url_link, search_key, user_agent)
        df_names = []
        df_prices = []
        df_comment_nums = []
        df_links = []

        # 打开模拟浏览器
        self.open_browser()
        # 进入目标JD网站
        self.browser.get(self.url)
        # 在浏览器输入目标商品名，然后搜索
        self.browser.find_element_by_id('key').send_keys(self.keyword)
        self.browser.find_element_by_class_name('button').click()

        # 开始爬取
        self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        print("################\n##开启数据爬虫##\n################\n")
        while self.isLastPage != True:
            page_num = 0
            names, prices, comment_nums, links, page_num = self.parse_JDpage()
            print("已爬取完第%s页" % page_num)
            df_names.extend(names)
            df_prices.extend(prices)
            df_comment_nums.extend(comment_nums)
            df_links.extend(links)
            self.turn_JDpage()

        # 退出浏览器
        self.browser.quit()

        # 保存结果
        results = pd.DataFrame({'title':df_names,
                                'price':df_prices,
                                'comment_num':df_comment_nums,
                                'url':df_links})
        results.to_csv(save_path, index = False)
        print("爬虫全部结束，共%d条数据，最终结果保存至%s" % (len(results),save_path))


if __name__ == '__main__':
    # 初始化参数配置
    url_link = 'https://www.jd.com'
    search_key = 'oppo find x2'
    user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
    save_path = r'/media/alvinai/Documents/comment_crawler/JD_results/Overview/JDoverview.csv'

    # 开启JD数据爬虫
    JDspider = Spider()
    JDspider.JDcrawl(url_link, search_key, user_agent, save_path)







