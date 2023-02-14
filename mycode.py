import pandas
import matplotlib.pyplot as plot
import pygal
import country_converter as toCode
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
articles['Citation'].groupby(articles['Year']).sum().plot(kind = 'bar')
plot.title("yearly_citation")
plot.xlabel("Year")
plot.ylabel("Citations")
plot.show()

#1.3 Number of publications across countries

countries = merged[['Article No.', 'Country']].drop_duplicates()
countries = countries.groupby(['Country'])['Article No.'].count().reset_index( name = 'Count').sort_values(['Count'], ascending = False)
world = pygal.maps.world.World()
world.title = 'Articles published by country'
theConverter = toCode.CountryConverter()
theCodes = theConverter.pandas_convert(series = countries['Country'], to = 'ISO2')
countries['countryCode'] = theCodes;
print(countries);
world.add('More than 10',countries.query('Count >10')['countryCode'].str.lower())
world.add('Between 5 and 10', countries.query('Count>10 and Count>=5')['countryCode'].str.lower())
world.add('Between 2 and 5', countries.query('Count <5 and Count>=2')['countryCode'].str.lower())
world.add('Just 1', countries.query('Count == 1')['countryCode'].str.lower())
world.render_to_png('world.png');


#1.4 Top 5 institutions with most published articles
new_df = merged[['Article No.', 'Author Affiliation']].drop_duplicates()
new_df = new_df.groupby(['Author Affiliation'])['Article No.'].count().reset_index( name = 'Count').sort_values(['Count'], ascending = False)
print(new_df[['Author Affiliation', 'Count']].reset_index(drop=True).head(n=5))

#1.5 Top 5 researchers that have the most h-index
top_researchers = authors[['Author Name','h-index']].sort_values(['h-index'], ascending = False).head(n=5).reset_index(drop=True)
print(top_researchers)

#Beginning of Question 2
