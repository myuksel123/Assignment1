import pandas

#read the files
authors = pandas.read_csv("authorinfo.csv")
articles = pandas.read_csv("articleinfo.csv")


#now, I will merge them
merged = authors.merge(articles, on = "Article No.")

#check if they merged well by printing it out in a file.
merged.to_csv("merged.csv", index = False)

