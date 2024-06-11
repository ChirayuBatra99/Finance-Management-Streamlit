from langchain.document_loaders import TextLoader
loader= TextLoader("nvda_news_1.txt")
data= loader.load()
# print(data[0].metadata)

from langchain.document_loaders.csv_loader import CSVLoader
loader1= CSVLoader("movies.csv")
data2= loader1.load()
# print((data2[1].metadata))

from langchain.document_loaders import UnstructuredURLLoader

loader = UnstructuredURLLoader(
    urls=[
        "https://www.moneycontrol.com/news/business/banks/hdfc-bank-re-appoints-sanmoy-chakrabarti-as-chief-risk-officer-11259771.html",
        "https://www.moneycontrol.com/news/business/markets/market-corrects-post-rbi-ups-inflation-forecast-icrr-bet-on-these-top-10-rate-sensitive-stocks-ideas-11142611.html"
    ]
)

data=loader.load()
# print(data)
# print(data[0].metadata)
# print(data[0].page_content[0:500])


text = """Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. 
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine. 
Set in a dystopian future where humanity is embroiled in a catastrophic blight and famine, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for humankind.

Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007 and was originally set to be directed by Steven Spielberg. 
Kip Thorne, a Caltech theoretical physicist and 2017 Nobel laureate in Physics,[4] was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar. 
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm. Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles. 
Interstellar uses extensive practical and miniature effects, and the company Double Negative created additional digital effects.

Interstellar premiered in Los Angeles on October 26, 2014. In the United States, it was first released on film stock, expanding to venues using digital projectors. The film received generally positive reviews from critics and grossed over $677 million worldwide ($715 million after subsequent re-releases), making it the tenth-highest-grossing film of 2014. 
It has been praised by astronomers for its scientific accuracy and portrayal of theoretical astrophysics.[5][6][7] Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades."""


print(text[0:100])
words= text.split(" ")
# print(words)
# print(len(words))

chunks=[]
s=""
for word in words:
    s += word + " "
    if len(s)>200:
        chunks.append(s)
        s = ""
        
chunks.append(s)
print(chunks[:1])

from langchain.text_splitter import CharacterTextSplitter
splitter= CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

# chunks= splitter.split_text(text)
# print(len(chunks))

# for chunk in chunks:
#     print(len(chunk))


from langchain.text_splitter import RecursiveCharacterTextSplitter

r_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
    chunk_size = 200,  # size of each chunk created
    chunk_overlap  = 0,  # size of  overlap between chunks in order to maintain the context
    length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
)

chunks = r_splitter.split_text(text)
for chunk in chunks:
    print(len(chunk))