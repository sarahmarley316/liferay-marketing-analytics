#How to web scrape information off of a webpage in Rstudio
library(‘rvest’)
url <- ’https://www.franchisegrade.com/search?page=1&results=2484'
webpage <- read_html(url)
title_data_html <- html_nodes(webpage,‘p.h4.search-name.text-center’)
title_data <- html_text(title_data_html)
head(title_data)

library(‘rvest’)
url <- ‘WEBSITE URL TO SCRAPE’
webpage <- read_html(url)
title_data_html <- html_nodes(webpage,‘p.text-center’)
title_data <- html_text(title_data_html)
head(title_data)
data <- matrix(title_data, ncol = 2)
colnames(data) <- c(“Franchise”,“Revenue Amount”)
options(max.print = 10000)
print(data)
