from nltk import RegexpTokenizer, RegexpParser, FreqDist
from nltk.corpus import stopwords
import pymorphy2
from pymystem3 import Mystem

tokenizer = RegexpTokenizer(r'\w+')
string = 'Значимость этих проблем настолько очевидна, что постоянный количественный рост и сфера нашей активности ' \
         'прекрасно подходит для реализации новых принципов формирования материально-технической и кадровой базы. Как ' \
         'уже неоднократно упомянуто, акционеры крупнейших компаний, которые представляют собой яркий пример ' \
         'континентально-европейского типа политической культуры, будут обнародованы. Идейные соображения высшего ' \
         'порядка, а также высококачественный прототип будущего проекта однозначно фиксирует необходимость ' \
         'направлений прогрессивного развития. '
grammar = "NP: {<ADJF>*<NOUN>|<NOUN>*<INFN>|<NPRO>*<INFN>}"
chunk_parser = RegexpParser(grammar)
morph = pymorphy2.MorphAnalyzer()
words = tokenizer.tokenize(string)
stop_words = stopwords.words('russian')
clean = [Mystem().lemmatize(word) for word in words if word not in stop_words]
l = []

for n in clean:
    t = (morph.parse(n[0])[0][0], morph.parse(n[0])[0].tag.POS)
    l.append(t)

tree = chunk_parser.parse(l)
tree.draw()
fd = FreqDist(l)
print(fd.most_common(5))
