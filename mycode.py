import pandas
import matplotlib.pyplot as plot
#Problem 1

#read the files
authors = pandas.read_csv("authorInfo.csv")
articles = pandas.read_csv("articleInfo.csv")


#now, I will merge them into one data frame
merged = authors.merge(articles, on = "Article No.")

#filling the empty cells with the value 0
merged.fillna(0)

#check if they merged well by printing it out in a file.
merged.to_csv("merged.csv", index = False)

#1.1 yearly_publication figure
#pandas.value_counts((articles['Year'])).plot.bar()
#articles['Year'].plot(kind = 'hist')
articles['Article No.'].groupby([articles['Year']]).count().plot(kind='bar')
plot.title("yearly_publication")
plot.xlabel("Year")
plot.ylabel("Articles Published")
plot.show()

#1.2 yearly_citation figure
