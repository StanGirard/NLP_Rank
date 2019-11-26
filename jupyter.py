#!/usr/bin/env python
# coding: utf-8

# ## Load Data From File

# In[91]:


import pickle
def read_corpus_from_file(filename):
        corpus = []
        with open(filename, 'rb') as fp:
            corpus = pickle.load(fp)
        return corpus
corpus = read_corpus_from_file("corpus.txt")
if not corpus:
    print("Corpus not loaded")
else:
    print("Corpus loaded")


# ## Normalize Text for Spacy

# In[115]:


import re
import nltk
from nltk.corpus import stopwords
stop = ['cliquez','devez','ça','pouvez','permet','très','être','mentions','dune','dit','com','min','bonjour','cest', 'jai','voir', 'co','avis','faut','dun','a','abord','afin','ah','ai','aie','ainsi','allaient','allo','allons','apres','assez','attendu','au','aucun','aucune','aujourd','aujourdhui','auquel','aura','auront','aussi','autre','autres','aux','auxquelles','auxquels','avaient','avais','avait','avant','avec','avoir','ayant','b','bah','beaucoup','bien','bigre','boum','bravo','brrr','c','ca','car','ce','ceci','cela','celle','celle-ci','celle','celles','celles-ci','celles-la','celui','celui-ci','celui-la','cent','cependant','certain','certaine','certaines','certains','certes','ces','cet','cette','ceux','ceux-ci','ceux-la','chacun','chaque','cher','chere','cheres','chers','chez','chiche','chut','ci','cinq','cinquantaine','cinquante','cinquantieme','cinquieme','clac','clic','combien','comme','comment','compris','concernant','contre','couic','crac','d','da','dans','de','debout','dedans','dehors','dela','depuis','derriere','des','desormais','desquelles','desquels','dessous','dessus','deux','deuxieme','deuxiemement','devant','devers','devra','different','differente','differentes','differents','dire','divers','diverse','diverses','dix','dix-huit','dixieme','dix-neuf','dix-sept','doit','doivent','donc','dont','douze','douzieme','dring','du','duquel','durant','e','effet','eh','elle','elle-meme','elles','elles-memes','en','encore','entre','envers','environ','es','est','et','etant','etaient','etais','etait','etant','etc','ete','etre','eu','euh','eux','eux-memes','excepte','f','façon','fais','faire','faisaient','faisant','fait','feront','fi','flac','floc','font','g','gens','h','ha','he','hein','helas','hem','hep','hi','ho','hola','hop','hormis','hors','hou','houp','hue','hui','huit','huitieme','hum','hurrah','i','il','ils','importe','j','je','jusqu','jusque','k','l','la','laquelle','las','le','lequel','les','lesquelles','lesquels','leur','leurs','longtemps','lorsque','lui','lui-meme','m','ma','maint','mais','malgre','me','meme','memes','merci','mes','mien','mienne','miennes','miens','mille','mince','moi','moi-meme','moins','mon','moyennant','n','na','ne','neanmoins','neuf','neuvieme','ni','nombreuses','nombreux','non','nos','notre','notres','nous','nous-memes','nul','o','o|','oh','ohe','ole','olle','on','ont','onze','onzieme','ore','ou','ouf','ouias','oust','ouste','outre','p','paf','pan','par','parmi','partant','particulier','particuliere','particulierement','pas','passe','pendant','personne','peu','peut','peuvent','peux','pff','pfft','pfut','pif','plein','plouf','plus','plusieurs','plutot','pouah','pour','pourquoi','premier','premiere','premierement','pres','proche','psitt','puisque','q','qu','quand','quant','quanta','quant-a-soi','quarante','quatorze','quatre','quatre-vingt','quatrieme','quatriemement','que','quel','quelconque','quelle','quelles','quelque','quelques','quelquun','quels','qui','quiconque','quinze','quoi','quoique','r','revoici','revoila','rien','s','sa','sacrebleu','sans','sapristi','sauf','se','seize','selon','sept','septieme','sera','seront','ses','si','sien','sienne','siennes','siens','sinon','six','sixieme','soi','soi-meme','soit','soixante','son','sont','sous','stop','suis','suivant','sur','surtout','t','ta','tac','tant','te','tel','telle','tellement','telles','tels','tenant','tes','tic','tien','tienne','tiennes','tiens','toc','toi','toi-meme','ton','touchant','toujours','tous','tout','toute','toutes','treize','trente','tres','trois','troisieme','troisiemement','trop','tsoin','tsouin','tu','u','un','une','unes','uns','v','va','vais','vas','ve','vers','via','vif','vifs','vingt','vivat','vive','vives','vlan','voici','voila','vont','vos','votre','votre','votres','vous','vous-memes','vu','w','x','y','z','zut']
stop_words = set(stopwords.words("french"))
stop_words_english = set(stopwords.words("english"))
stop_words_multilingual = stop_words.union(stop_words_english)
new_stopwords_list = stop_words_multilingual.union(stop)

def normalize_text_list( textArray):
        corpus = []
        for i in range(len(textArray)):
            #Remove accents
            text = re.sub("'","  ", textArray[i])

            text = re.sub('"'," ", textArray[i])

            # Removes urls
            text = re.sub("/(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])/igm", " ", text)

            #text = str(unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore'))

            #Remove Special characters
            #text = re.sub('[^a-zA-Z0-9]', ' ', text)
            text = re.sub("[^a-zA-Z0-9áàéèíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇûç:.' ]", ' ', text)
            
            #Convert to lowercase
            text = text.lower()
            
            #remove tags
            text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
            
            # remove special characters and digits
            text=re.sub("(\\d|\\W)+"," ",text)
            
            ##Convert to list from string
            text = text.split()
            
            ##Stemming
            #ps=PorterStemmer()
            #Lemmatisation
            #lem = WordNetLemmatizer()
            
            text = " ".join(text)
            corpus.append(text)
        return corpus
normalized = normalize_text_list(corpus)
separator = ' '
joined = separator.join(normalized)


# In[116]:


#import nltk
#from nltk.tokenize import word_tokenize
#from nltk.tag import pos_tag

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
#joined_processed = preprocess(joined) 


# # Spacy

# In[117]:


import spacy
from spacy import displacy
from collections import Counter

import fr_core_news_md
nlp = fr_core_news_md.load()

article = nlp(joined)
print(len(article.ents))


# In[118]:


labels = [x.label_ for x in article.ents]
Counter(labels)


# In[119]:


items = [x.text for x in article.ents]
items = [word for word in items if not word in  
                    new_stopwords_list] 

Counter(items).most_common(40)


# In[ ]:





# In[ ]:




