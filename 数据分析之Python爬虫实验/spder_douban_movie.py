import urllib.request as ur
import re

headers = ("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36")
opener = ur.build_opener()
opener.addheaders = [headers]
ur.install_opener(opener)

urlorg = "https://movie.douban.com/subject/26213252/reviews?start="

for i in range(0, 100, 20):
    url = urlorg + str(i)
    every_page = ur.urlopen(url).read().decode("utf-8")
    pat = 'data-rid="(.*?)" title="有用"'
    review_id = re.compile(pat).findall(every_page)

    for j in range(0, 20):
        review_url = "https://movie.douban.com/review/" + str(review_id[j])
        review = ur.urlopen(review_url).read().decode("utf-8")
        pat2 = '<meta name="description" content="(.*?)" />'
        pat3 = 'data-original="1">(.*?)<div class="copyright">'
        review_title = re.compile(pat2).findall(review)
        review_content = re.compile(pat3, re.S).findall(review)
        print(review_title)
        print("-----------------------------")
        print(review_content)

        fh = open("./douban_movie_review.html", "a")
        fh.write(str(review_title))
        fh.write(str(review_content))
        fh.close()
