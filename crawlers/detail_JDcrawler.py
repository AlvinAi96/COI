from selenium import webdriver
import selenium
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import time
from selenium.webdriver.common.keys import Keys
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class DetailSpider():
    def clean_overview(self, csv_path, search_keyword):
        '''
            清洗数据
                1. 清洗掉少于11条评论的数据
                2. 清洗掉含‘拍拍’关键词的数据
                3. 清洗掉不含搜索关键词的数据
                4. 清洗掉价格过低的数据

            输入：
                1. csv_path                (str): 爬取的overview商品展示页下的结果文件路径。
                2. search_keyword          (str): 爬取的overview商品展示页下搜索关键词。
            输出：
                1. overview_df    (pd.DataFrame)： 清洗好的overview数据
        '''
        overview_path = csv_path
        overview_df = pd.read_csv(overview_path)
        search_key = search_keyword

        # 1. 清洗掉少于11条评论的数据
        print("原始数据量：%s" % len(overview_df))
        drop_idxs = []
        comment_nums = overview_df['comment_num']
        for i, comment_num in enumerate(comment_nums):
            try:
                if int(comment_num[:-3]) in list(range(11)):
                    drop_idxs.append(i)
            except:
                pass
        print("清洗掉少于11条评论的数据后的数据量：%s（%s-%s）" % (len(overview_df) - len(drop_idxs), len(overview_df), len(drop_idxs)))
        overview_df.drop(drop_idxs, axis = 0, inplace = True)
        overview_df = overview_df.reset_index(drop = True)

        # 2. 清洗掉含‘拍拍’关键词的数据
        drop_idxs = []
        comment_titles = overview_df['title']
        for i, title in enumerate(comment_titles):
            try:
                if title.startswith('拍拍'):
                    drop_idxs.append(i)
            except:
                pass
        print("清洗掉含‘拍拍’关键词的数据后的数据量：%s（%s-%s）" % (len(overview_df) - len(drop_idxs), len(overview_df), len(drop_idxs)))
        overview_df.drop(drop_idxs, axis = 0, inplace = True)
        overview_df = overview_df.reset_index(drop = True)

        # 3. 清洗掉不含搜索关键词的数据
        drop_idxs = []
        comment_titles = overview_df['title']
        for i, title in enumerate(comment_titles):
            if search_key.replace(" ","") not in title.lower().replace(" ",""):
                drop_idxs.append(i)
        print("清洗掉不含搜索关键词的数据后的数据量：%s（%s-%s）" % (len(overview_df) - len(drop_idxs), len(overview_df), len(drop_idxs)))
        overview_df.drop(drop_idxs, axis = 0, inplace = True)
        overview_df = overview_df.reset_index(drop = True)

        # 4. 清洗掉价格过低/过高的数据
        drop_idxs = []
        comment_prices = overview_df['price']
        prices_df = {}
        for p in comment_prices:
            if p not in list(prices_df.keys()):
                prices_df[p] = 1
            else:
                prices_df[p] += 1
        # print("各价格下的商品数：", prices_df)
        # {4499: 89, 5999: 5, 6099: 1, 12999: 2, 6999: 1, 89: 1, 29: 1}
        # 通过上述结果，我们只要价位为4499的商品结果即可了
        for i, p in enumerate(comment_prices):
            if p != 4499.0:
                drop_idxs.append(i)
        print("清洗掉价格过低/过高的数据后的数据量：%s（%s-%s）" % (len(overview_df) - len(drop_idxs), len(overview_df), len(drop_idxs)))
        overview_df.drop(drop_idxs, axis = 0, inplace = True)
        overview_df = overview_df.reset_index(drop = True)
        return overview_df


    def open_browser(self):
        '''设置浏览器'''
        # 若下列命令报错，请进入下面链接下载chromedriver然后放置在/user/bin/下即可
        # https://chromedriver.storage.googleapis.com/index.html?path=2.35/
        self.options = webdriver.ChromeOptions()
        self.browser = webdriver.Chrome(options = self.options)
        # 隐式等待：等待页面全部元素加载完成（即页面加载圆圈不再转后），才会执行下一句，如果超过设置时间则抛出异常
        try:
            self.browser.implicitly_wait(50)
        except:
            print("页面无法加载完成，无法开启爬虫操作！")
        # 显式等待：设置浏览器最长允许超时的时间
        self.wait = WebDriverWait(self.browser, 30)


    def init_variable(self, csv_path, search_key, user_agent):
        '''初始化变量'''
        self.csv_path = csv_path # 商品总览页爬取结果文件路径
        self.keyword = search_key # 商品搜索关键词
        self.isLastPage = False # 是否为页末
        self.ignore_page = False # 是否进入到忽略评论页面
        self.user_agent = user_agent # 用户代理，这里暂不用它
        print("###############\n##初始化参数##\n###############\ncsv_path：%s\nkeyword：%s\nuser_agent：%s\n\n" % (self.csv_path, self.keyword, self.user_agent))


    def parse_JDpage(self):
        try:
            time.sleep(10) # 下拉后等10s
            # 定位元素（用户名，用户等级，用户评分，用户评论，评论创建时间，购买选择，页码）
            user_names = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="user-info"]')))
            user_levels = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="user-level"]')))
            user_stars = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="comment-column J-comment-column"]/div[starts-with(@class, "comment-star")]')))
            comments = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="comment-column J-comment-column"]/p[@class="comment-con"]')))
            order_infos = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="comment-item"]//div[@class="order-info"]')))
            if self.ignore_page == False:
                # 如果没进入忽略页
                page_num = self.wait.until(EC.presence_of_element_located((By.XPATH, '//a[@class="ui-page-curr"]')))
            else:
                # 如果进入忽略页
                page_num = self.wait.until(EC.presence_of_element_located((By.XPATH, '//div[@class="ui-dialog-content"]//a[@class="ui-page-curr"]')))
            # 获取元素下的字段值
            user_names = [user_name.text for user_name in user_names]
            user_levels = [user_level.text for user_level in user_levels]
            user_stars = [user_star.get_attribute('class')[-1] for user_star in user_stars]
            create_times = [" ".join(order_infos[0].text.split(" ")[-2:]) for order_info in order_infos]
            order_infos = [" ".join(order_infos[0].text.split(" ")[:-2]) for order_info in order_infos]
            comments = [comment.text for comment in comments]
            page_num = page_num.text
        except selenium.common.exceptions.TimeoutException:
            print('parse_page: TimeoutException 网页超时')
            self.browser.refresh()
            self.browser.find_element_by_xpath('//li[@data-tab="trigger" and @data-anchor="#comment"]').click()
            time.sleep(30)
            user_names, user_levels, user_stars, comments, create_times, order_infos, page_num = self.parse_JDpage()
        except selenium.common.exceptions.StaleElementReferenceException:
            print('turn_page: StaleElementReferenceException 某元素因JS刷新已过时没出现在页面中')
            user_names, user_levels, user_stars, comments, create_times, order_infos, page_num = self.parse_JDpage()
        return user_names, user_levels, user_stars, comments, create_times, order_infos, page_num


    def turn_JDpage(self):
        # 移到页面末端并点击‘下一页’
        try:
            if self.ignore_page == False:
                self.browser.find_element_by_xpath('//a[@class="ui-pager-next" and @clstag]').send_keys(Keys.ENTER)
            else:
                self.browser.find_element_by_xpath('//a[@class="ui-pager-next" and @href="#none"]').send_keys(Keys.ENTER)
            time.sleep(3) # 点击完等3s
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(5) # 下拉后等5s
        # 如果找不到元素
        except selenium.common.exceptions.NoSuchElementException:
            if self.ignore_page == False:
                try:
                    # 如果有忽略评论的页面但没进入，则进入继续翻页
                    self.browser.find_element_by_xpath('//div[@class="comment-more J-fold-comment hide"]/a').send_keys(Keys.ENTER)
                    self.ignore_page = True
                    print("有忽略评论的页面")
                except:
                    # 如果没有忽略评论的页面且最后一页
                    print("没有忽略评论的页面")
                    self.ignore_page = True
                    self.isLastPage = True
            else:
                # 如果有忽略评论的页面且到了最后一页
                print("没有忽略评论的页面")
                self.isLastPage = True
        except selenium.common.exceptions.TimeoutException:
            print('turn_page: TimeoutException 网页超时')
            time.sleep(30)
            self.turn_JDpage()
        # 如果因为JS刷新找不到元素，重新刷新
        except selenium.common.exceptions.StaleElementReferenceException:
            print('turn_page: StaleElementReferenceException 某元素因JS刷新已过时没出现在页面中')
            self.turn_JDpage()


    def JDcrawl_detail(self, csv_path, search_key, user_agent):
        # 初始化参数
        self.init_variable(csv_path, search_key, user_agent)
        unfinish_crawls = 0 # 记录因反爬虫而没有完全爬取的商品数
        # 清洗数据
        self.overview_df = self.clean_overview(self.csv_path, self.keyword)

        # 依次进入到单独的商品链接里去
        for url in tqdm(list(self.overview_df['url'][3:])):
            df_user_names = []
            df_user_levels = []
            df_user_stars = []
            df_comments = []
            df_create_times = []
            df_order_infos = []

            # 打开模拟浏览器
            self.open_browser()
            # 进入目标网站
            self.browser.get(url)
            time.sleep(35)
            # 进入评论区
            self.browser.find_element_by_xpath('//li[@data-tab="trigger" and @data-anchor="#comment"]').click()
            time.sleep(15)
            # 开始爬取
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            self.isLastPage = False
            self.ignore_page = False
            self.lastpage = 0
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " 开启数据爬虫url:",url)
            while self.isLastPage != True:
                page_num = 0
                user_names, user_levels, user_stars, comments, create_times, order_infos, page_num = self.parse_JDpage()
                # 如果某页因为反爬虫无法定位到‘下一页’元素，导致重复爬取同一页，则保留之前的爬取内容，然后就不继续爬这个商品了
                if self.lastpage != page_num:
                    self.lastpage = page_num
                    print("已爬取完第%s页" % page_num)
                    df_user_names.extend(user_names)
                    df_user_levels.extend(user_levels)
                    df_user_stars.extend(user_stars)
                    df_comments.extend(comments)
                    df_create_times.extend(create_times)
                    df_order_infos.extend(order_infos)
                    self.turn_JDpage()
                else:
                    unfinish_crawls += 1
                    self.browser.quit()
                    break
            # 退出浏览器
            self.browser.quit()

            # 保存结果
            results = pd.DataFrame({'user_names':df_user_names,
                                    'user_levels':df_user_levels,
                                    'user_stars':df_user_stars,
                                    'omments':df_comments,
                                    'create_times':df_create_times,
                                    'order_infos':df_order_infos})
            url_id = url.split('/')[-1].split('.')[0]
            save_path = r'/media/alvinai/Documents/comment_crawler/JD_results/Detail/' + str(url_id) + '.csv'
            results.to_csv(save_path, index = False)
            print("爬虫结束，共%d条数据，结果保存至%s" % (len(results),save_path))


if __name__ == '__main__':
    # 初始化参数配置
    csv_path = r'/media/alvinai/Documents/comment_crawler/JD_results/Overview/JDoverview.csv'
    search_key = 'oppo find x2'
    user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
    # save_path = r'/media/alvinai/Documents/comment_crawler/JD_results/Overview/JDdetails.csv'

    # 开启JD数据爬虫
    stime = time.time()
    JDspider = DetailSpider()
    JDspider.JDcrawl_detail(csv_path, search_key, user_agent)
    etime = time.time()
    cost_hrs = (etime - stime)/3600
    print('爬虫共计：%.3f' % cost_hrs)
