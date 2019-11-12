import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import unicodedata

stop_words = set(stopwords.words("french"))
test = str(unicodedata.normalize('NFKD', re.sub("'"," ", "j'ai c'est balîses")).encode('ASCII', 'ignore'))
tt = re.sub('[^a-zA-Z0-9]', ' ', test)
t = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",tt)
text = re.sub("(\\d|\\W)+"," ",t)
print(text)
text = text.split()

ps=PorterStemmer()
lem = WordNetLemmatizer()
text = [word for word in text if not word in  
                    stop_words] 
text = " ".join(text)
print(text)