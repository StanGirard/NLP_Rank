# -*- coding: utf-8 -*-
import requests
import urllib.request
from bs4 import BeautifulSoup
from http.client import HTTPSConnection
from base64 import b64encode
from json import loads
from json import dumps
from random import Random
from inscriptis import get_text
from client import RestClient
from test import RecursiveScraper
import unicodedata
import nltk
import re
import os.path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from scipy.sparse import coo_matrix
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import numpy
from gensim import corpora
from gensim.models import LsiModel
import pickle

stop = ['mentions','dune','dit','com','min','bonjour','cest', 'jai','voir', 'co','avis','faut','dun','a','abord','afin','ah','ai','aie','ainsi','allaient','allo','allons','apres','assez','attendu','au','aucun','aucune','aujourd','aujourdhui','auquel','aura','auront','aussi','autre','autres','aux','auxquelles','auxquels','avaient','avais','avait','avant','avec','avoir','ayant','b','bah','beaucoup','bien','bigre','boum','bravo','brrr','c','ca','car','ce','ceci','cela','celle','celle-ci','celle','celles','celles-ci','celles-la','celui','celui-ci','celui-la','cent','cependant','certain','certaine','certaines','certains','certes','ces','cet','cette','ceux','ceux-ci','ceux-la','chacun','chaque','cher','chere','cheres','chers','chez','chiche','chut','ci','cinq','cinquantaine','cinquante','cinquantieme','cinquieme','clac','clic','combien','comme','comment','compris','concernant','contre','couic','crac','d','da','dans','de','debout','dedans','dehors','dela','depuis','derriere','des','desormais','desquelles','desquels','dessous','dessus','deux','deuxieme','deuxiemement','devant','devers','devra','different','differente','differentes','differents','dire','divers','diverse','diverses','dix','dix-huit','dixieme','dix-neuf','dix-sept','doit','doivent','donc','dont','douze','douzieme','dring','du','duquel','durant','e','effet','eh','elle','elle-meme','elles','elles-memes','en','encore','entre','envers','environ','es','est','et','etant','etaient','etais','etait','etant','etc','ete','etre','eu','euh','eux','eux-memes','excepte','f','façon','fais','faire','faisaient','faisant','fait','feront','fi','flac','floc','font','g','gens','h','ha','he','hein','helas','hem','hep','hi','ho','hola','hop','hormis','hors','hou','houp','hue','hui','huit','huitieme','hum','hurrah','i','il','ils','importe','j','je','jusqu','jusque','k','l','la','laquelle','las','le','lequel','les','lesquelles','lesquels','leur','leurs','longtemps','lorsque','lui','lui-meme','m','ma','maint','mais','malgre','me','meme','memes','merci','mes','mien','mienne','miennes','miens','mille','mince','moi','moi-meme','moins','mon','moyennant','n','na','ne','neanmoins','neuf','neuvieme','ni','nombreuses','nombreux','non','nos','notre','notres','nous','nous-memes','nul','o','o|','oh','ohe','ole','olle','on','ont','onze','onzieme','ore','ou','ouf','ouias','oust','ouste','outre','p','paf','pan','par','parmi','partant','particulier','particuliere','particulierement','pas','passe','pendant','personne','peu','peut','peuvent','peux','pff','pfft','pfut','pif','plein','plouf','plus','plusieurs','plutot','pouah','pour','pourquoi','premier','premiere','premierement','pres','proche','psitt','puisque','q','qu','quand','quant','quanta','quant-a-soi','quarante','quatorze','quatre','quatre-vingt','quatrieme','quatriemement','que','quel','quelconque','quelle','quelles','quelque','quelques','quelquun','quels','qui','quiconque','quinze','quoi','quoique','r','revoici','revoila','rien','s','sa','sacrebleu','sans','sapristi','sauf','se','seize','selon','sept','septieme','sera','seront','ses','si','sien','sienne','siennes','siens','sinon','six','sixieme','soi','soi-meme','soit','soixante','son','sont','sous','stop','suis','suivant','sur','surtout','t','ta','tac','tant','te','tel','telle','tellement','telles','tels','tenant','tes','tic','tien','tienne','tiennes','tiens','toc','toi','toi-meme','ton','touchant','toujours','tous','tout','toute','toutes','treize','trente','tres','trois','troisieme','troisiemement','trop','tsoin','tsouin','tu','u','un','une','unes','uns','v','va','vais','vas','ve','vers','via','vif','vifs','vingt','vivat','vive','vives','vlan','voici','voila','vont','vos','votre','votre','votres','vous','vous-memes','vu','w','x','y','z','zut']



class GetList:
    client = RestClient("contact@twodroppoint.com", "2iHUO8lhRFD5vqEP")
    stop_words = set(stopwords.words("french"))
    stop_words_english = set(stopwords.words("english"))
    stop_words_multilingual = stop_words.union(stop_words_english)
    new_stopwords_list = stop_words_multilingual.union(stop)
    def create_request(self, post_data):
        response = self.client.post("/v2/live/srp_tasks_post", dict(data=post_data))
        if response["status"] == "error":
            print("error. Code: %d Message: %s" % (response["error"]["code"], response["error"]["message"]))
            return []
        else:
            return response

    def get_results_from_keywords(self, keywords):
        rnd = Random() #you can set as "index of post_data" your ID, string, etc. we will return it with all results.
        post_data = dict()
        post_data[rnd.randint(1, 30000000)] = dict(
        se_name="google.fr",
        se_language="French",
        se_localization= 'fr-fr',
        loc_id= 1006094,
        key=keywords
        )
        return self.create_request(post_data)

    def get_results_from_taskid(self, taskId):
        srp_response = self.client.get("/v2/srp_tasks_get/%d" % (taskId))
        if srp_response["status"] == "error":
            print("error. Code: %d Message: %s" % (srp_response["error"]["code"], srp_response["error"]["message"]))
            return []
        else:
            return srp_response
    def get_results(self, keywords):
        allinfo = []
        if (type(keywords) is int):
            allinfo = self.get_results_from_taskid(keywords)
        else:
            allinfo = self.get_results_from_keywords(keywords)
        return allinfo

    def extract_url_from_results(self, data):
        url_result = []
        for i in data["results"]["organic"]:
            url_result.append(i["result_url"])
        return url_result
    
    def extract_text_from_url(self, list):
        text = []
        count = 0
        for i in list:
            try:
                html = urllib.request.urlopen(i, timeout=3).read().decode('utf-8')
                text.append(get_text(html))
                count += 1
            except:
                print("Error: " + i)
        return text

    def normalize_text_list(self, textArray):
        corpus = []
        for i in range(len(textArray)):
            #Remove accents
            text = re.sub("'"," ", textArray[i])

            text = re.sub('"'," ", text)

            # Removes urls
            text = re.sub("/(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])/igm", " ", text)

            text = str(unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore'))

            #Remove Special characters
            text = re.sub('[^a-zA-Z0-9]', ' ', text)
            
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
            text = [word for word in text if not word in  
                    self.new_stopwords_list] 
            text = " ".join(text)
            corpus.append(text)
        return corpus
    def write_corpus_to_file(self,corpus):
        with open("corpus.txt", "wb") as fp:
            pickle.dump(corpus, fp)

    def write_corpus_to_file_from_keywords(self, keywords):
        allinfo = self.get_results(keywords)
        vim = self.extract_url_from_results(allinfo)
        tex = self.extract_text_from_url(vim)
        corpus = self.normalize_text_list(tex)
        self.write_corpus_to_file(corpus)
    def read_corpus_from_file(self, filename):
        corpus = []
        with open(filename, 'rb') as fp:
            corpus = pickle.load(fp)
        return corpus

    def get_top_n_ygrams_words(self, corpus, n=None, y=1):
        vec = CountVectorizer(min_df = 0.1,stop_words=self.new_stopwords_list, max_features=10000, ngram_range=(y,y)).fit(corpus)
        bag_of_words  = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in      
                    vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], 
                        reverse=True)
        frame =  pd.DataFrame(words_freq[:n])
        switcher = {
        1: "Monogram",
        2: "Bigram",
        3: "Trigram",
        }
        switcherFreq = {
        1: "Freq_Mono",
        2: "Freq_Bi",
        3: "Freq_Tri",
        }
        frame.columns = [switcher.get(y), switcherFreq.get(y)]
        return frame
    
    def get_all_ygrams(self,corpus, monograms, bigrams, trigrams):
        mono = self.get_top_n_ygrams_words(corpus, monograms, 1)
        bi = self.get_top_n_ygrams_words(corpus, bigrams, 2)
        tri = self.get_top_n_ygrams_words(corpus, trigrams, 3)
        monodict = mono.to_dict()
        bidict = bi.to_dict()
        tridict = tri.to_dict()
        all = {}
        all.update(monodict)
        all.update(bidict)
        all.update(tridict)
        return all
    
    def prepare_corpus(self,doc_clean):
        """
        Input  : clean document
        Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
        Output : term dictionary and Document Term Matrix
        """
        # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
        dictionary = corpora.Dictionary(doc_clean)
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        # generate LDA model
        return dictionary,doc_term_matrix
    def create_gensim_lsa_model(self,doc_clean,number_of_topics,words):
        """
        Input  : clean document, number of topics and number of words associated with each topic
        Purpose: create LSA model using gensim
        Output : return LSA model
        """
        dictionary,doc_term_matrix=self.prepare_corpus(doc_clean)
        # generate LSA model
        lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
        return lsamodel
    
    def compute_coherence_values(self,dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
        """
        Input   : dictionary : Gensim dictionary
                corpus : Gensim corpus
                texts : List of input texts
                stop : Max num of topics
        purpose : Compute c_v coherence for various number of topics
        Output  : model_list : List of LSA topic models
                coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, stop, step):
            # generate LSA model
            print("Number of topics: ", num_topics)
            model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
            coherenceSS = coherencemodel.get_coherence()
            coherence_values.append(coherenceSS)
        return model_list, coherence_values
    
    def plot_graph(self, doc_clean,start, stop, step):
        dictionary,doc_term_matrix=self.prepare_corpus(doc_clean)
        
        model_list, coherence_values = self.compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                                stop, start, step)
        # Show graph
        
        x = range(start, stop, step)
        for i in range(len(coherence_values)):
            if float(coherence_values[i]) > float("0.4"):
                print(coherence_values[i])
        
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()
    def calculate_optimal_coherence_value(self, doc_clean, start, stop, step):
        dictionary,doc_term_matrix=self.prepare_corpus(doc_clean)
        model_list, coherence_values = self.compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                                stop, start, step)
        max = 0
        topics = 0

        for i in range(len(coherence_values)):
            if float(coherence_values[i]) > float("0.4"):
                if coherence_values[i] > max:
                    topics = i
                print(coherence_values[i])
        return topics
    def generate_optimal_topic(self, doc_clean, start, stop, step):
        topics = self.calculate_optimal_coherence_value(doc_clean, start, stop, step)
        print(self.create_gensim_lsa_model(doc_clean,topics,3))
    
    def corpus_to_gensim(self,corpus):
        gensim = []
        for i in corpus:
            token = i.split()
            gensim.append(token)
        return gensim

    def generate_results(self, keywords):
        allinfo = self.get_results(keywords)
        vim = self.extract_url_from_results(allinfo)
        tex = self.extract_text_from_url(vim)
        corpus = self.normalize_text_list(tex)
        gensim = self.corpus_to_gensim(corpus)
        model = self.create_gensim_lsa_model(gensim,10,3)
        #start,stop,step = 2,12,1
        #self.plot_graph(gensim,start,stop,step)
        results = self.get_all_ygrams(corpus, 40, 20, 10)
        print(results)
        return results

    def generate_results_from_file(self, filename):
        corpus = self.read_corpus_from_file(filename)
        gensim = self.corpus_to_gensim(corpus)
        #model = self.create_gensim_lsa_model(gensim,10,3)
        start,stop,step = 2,10,1

        self.generate_optimal_topic(gensim,start,stop,step)
        #results = self.get_all_ygrams(corpus, 40, 20, 10)
        #print(results)
        return 1

    def generate_results_from_url(self, url):
        rscraper = RecursiveScraper(url)
        rscraper.scrape()
        rscraper.urls
        tex = self.extract_text_from_url(rscraper.urls)
        corpus = self.normalize_text_list(tex)
        gensim = self.corpus_to_gensim(corpus)
        #model = self.create_gensim_lsa_model(gensim,10,3)
        start,stop,step = 2,10,1

        self.generate_optimal_topic(gensim,start,stop,step)
        results = self.get_all_ygrams(corpus, 40, 20, 10)
        print(results)
        return results


def main():
    s = GetList()
    s.generate_results_from_url("https://epita.fr")
    #s.write_corpus_to_file_from_keywords("Comment optimiser son site")
    #s.generate_results_from_file("corpus.txt")

if __name__ == "__main__":
    main()


#s.write_corpus_to_file_from_keywords("Comment optimiser son site")




    