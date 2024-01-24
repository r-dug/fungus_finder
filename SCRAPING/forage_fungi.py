#can I just load a model and fit to
import os
import time
required_packages = ["requests", "fake_useragent","selenium", "webdriver_manager", "csv"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        os.system(f"pip install {package}")

        os.system(f"pip install {package}")
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import requests

def fungal_lists():
    fungusList = []
    pathList = []

    # # this loop cleaned up superfluous "scraped" directories.
    # # we no longer want this to happen, but it's kind of good code snippet to 
    # to let hang around.
    for root,dirs,files in os.walk("./fungi/",topdown=True):
        for dir in dirs:
            if not "not_fungi" in root:
                
                if "scraped" in root:
                    os.system(f"rm -r {root}/{dir}")

    for root,dirs,files in os.walk("./fungi/",topdown=True):
        for dir in dirs:

            if not "not_fungi" in root and dir != "not_fungi" and dir != "scraped":
                fungidir = (f"{os.path.join(root,dir)}/scraped")
                pathList.append(fungidir)
                fungus = "+".join(dir.split("_")[-2:])
                fungusList.append(fungus)
    return fungusList, pathList


def write_bookmark(fungus):
    with open("bookmark.txt", "w") as f:
        f.write(f"{fungus}\n")

def read_bookmark(pathList):
    bookmark = []
    with open("bookmark.txt", "r") as f:
        i = 0
        for line in f.readlines():
            if i == 0:
                bookmark.append(line.strip('\n'))
            else:
                bookmark.append(line)
            i+=1
    fungus_idx = pathList.index(bookmark[0])
    bookmark[0] = fungus_idx
    return bookmark
    

def is_end_of_page(driver):
    try:
        end_element = driver.find_element(By.TAG_NAME, 'footer')
        if end_element:
            return True
    except:
        return False
    
                
def find_images_src(driver, images_src_list):
    parent = driver.find_element(By.XPATH, '//*[@id="islrg"]/div[1]')
    child_images = parent.find_elements(By.TAG_NAME, 'img')
    src_list = []
    for child_image in child_images:
        image_src = child_image.get_attribute('src')
        if image_src != None:
            if "https://" in image_src and "images" in image_src and image_src not in images_src_list:
                src_list.append(image_src)
    return src_list


def dl_image(src, dir, fungus):
    response = requests.get(src)
    try:
        with open(f"{dir}/{fungus}.jpg", "xb") as f:
            f.write(response.content)
    except Exception as e:
        print(e)


def main():
    ua = UserAgent()
    user_agent = ua.random
    options = Options()
    # options.add_argument('--headless')
    options.add_argument(f'user-agent={user_agent}')
    driver = webdriver.Chrome(options=options)
    google = "https://www.google.com"
    fungusList, pathList = fungal_lists()
    for path in pathList:
        print(path)
    bookmark  = read_bookmark(pathList)
    print(bookmark)

    for path in pathList[bookmark[0]:]:
        print(path)
        try:
            os.system(f"mkdir {path}")
        except Exception as e:
            print(e)

    for i in range(bookmark[0],len(fungusList)):
        images_src_list = []
        fungus = fungusList[i]
        dir = f'/home/richard/Desktop/test/{fungus}'
        dir2 = pathList[i]
        print(f"{fungus}\n{dir2}")
        driver.get(f"{google}/search?q={fungus}&tbm=isch")
        timeout = 0
        while len(images_src_list)<200 and timeout < 30:
            time.sleep(2)
            new_src = find_images_src(driver, images_src_list)
            for src in new_src:
                images_src_list.append(src)
            driver.execute_script("window.scrollBy(0, 500);")
            timeout += 1
            
        num_images = 0
        for src in images_src_list:
            # user_agent = ua.random
            # options.add_argument(f'user-agent={user_agent}')
            # driver = webdriver.Chrome(options=options)
            fungus = fungus.replace("+","_")
            try:
                driver.get(src)
                dl_image(src,dir2,f"{fungus}{num_images}")
                num_images+=1
            except Exception as e:
                print(e)
                print(f"failure for: {fungus}")
        write_bookmark(pathList[i+1])
        bookmark[0]+=1
        print(f"{bookmark[0]}/{len(pathList)}")

main()
