
import json
import unicodedata
import math
import random
import numpy as np
import gensim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from gensim.models import Word2Vec
from gensim.models import phrases
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

use_cuda = torch.cuda.is_available()


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
            
    def addSentence(self, sentence):
        try:
            #f=open("w2v_"+self.name,"r")
            model=gensim.models.KeyedVectors.load_word2vec_format("w2v_"+self.name)
            weights=model.syn0
        except FileNotFoundError:
            print(len(sentence))
            ph = phrases.Phrases(sentence)
            bigram_transformer = phrases.Phraser(ph)
            trigram=phrases.Phrases(bigram_transformer[sentence])
            ngram = phrases.Phrases(trigram[sentence])
            #ngram=phrases.Phrases(trigram[bigram_transformer[sentence]])
            model = Word2Vec(ngram[trigram[bigram_transformer[sentence]]], size=40000, window=5, min_count=1, workers=4, sg=0, iter=80)
            model.wv.save_word2vec_format("w2v_"+self.name)
            #print(sentence[1:10])
                #print("Fresh :",model["fresh"])
            #print("ताजा :",model["ताजा"])
            weights = model.wv.syn0
        #print(weights)
        np.save(open("embed"+self.name+".txt", 'wb'), weights)

        vocab = dict([(k, v.index) for k, v in model.vocab.items()])
        with open("vocab"+self.name+".txt", 'w',encoding='utf-8') as f:
            f.write(json.dumps(vocab))
        with open("vocab"+self.name+".txt", 'r',encoding='utf-8') as f:
            data = json.loads(f.read())
        self.word2index = data
        self.index2word = dict([(v, k) for k, v in data.items()])
        self.n_words = len(model.vocab)

        print(self.name+":", self.n_words)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('dataset/training_set1.txt' ,encoding='utf-8').read().strip().split('\n')
    
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    lines = open('dataset/test_set1.txt' ,encoding='utf-8').read().strip().split('\n')
    test = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse is True:
        
        #print(reverse)
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs, test


MAX_LENGTH = 21


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def corpus_partition(count, pairs):
    sno=[i for i in range(1,count+1)]
    random.shuffle(sno)
    random.shuffle(sno)
    p,t, i=[],[],0
    for pair in pairs:
        if i in sno[:math.ceil(0.9*count)+1]:
            p.append(pair)
        else:
            t.append(pair)
        i=i+1
    return p,t
def prepareData(lang1, lang2, reverse=False):
    #print(reverse)
    input_lang, output_lang, pairs, test = readLangs(lang1, lang2, reverse)
    #print("%s" % input_lang.name)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    test=filterPairs(test)
    print("Trimmed to %s training pairs" % len(pairs))
    print("Trimmed to %s testing pairs" % len(test))
    print("Counting words...")
    t=[]
    h=[]

    for pair in pairs:
        t.append(pair[0].split(" "))
        h.append(pair[1].split(" "))

    for t1 in test:
        t.append(t1[0].split(" "))
        h.append(t1[1].split(" "))
    input_lang.addSentence(h)
    output_lang.addSentence(t)

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print (len(pairs),len(test))
    size=len(pairs)
    return input_lang, output_lang, pairs, test, size


input_lang, output_lang, pairs, test, size = prepareData('hin', 'tam', False)
print(random.choice(pairs))



class AttnEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1,dropout_p=0.1):
        super(AttnEncoderRNN, self).__init__()
        self.input_size=input_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_p=dropout_p

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.hidden = self.initHidden()

    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)))

        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        result = (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
        #Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.hidden = self.initHidden()

    def init_hidden(self):
        return (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
        
    def forward(self, input1, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input1).view(1, 1, -1)
        embedded = self.dropout(embedded)
        #print(embedded[0].size(), hidden[0].size())
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0][0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
        #Variable(torch.zeros(1, 1, self.hidden_size))
        

def index4word(lang,word):
    try:
        return lang.word2index[word]
    except KeyError:
        return 0

def indexesFromSentence(lang, sentence):
    return [index4word(lang,word) for word in sentence.split(" ")]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)

    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)



teacher_forcing_ratio = 0.5

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder1_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
   
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]
        encoder_output, encoder_hidden = encoder(encoder_outputs[ei], encoder1_hidden)
        encoder_outputs[ei]=encoder_output[0][0]
        

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            
            loss += criterion(decoder_output[0], target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainEpochs(encoder, decoder, n_epochs, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(size)]
    criterion = nn.NLLLoss()
        
    for epoch in range(1, n_epochs + 1):
        start1 = time.strftime("%H:%M:%S")
        for i in range(0,size):
            training_pair = training_pairs[i]
            input_variable = training_pair[0]
            target_variable = training_pair[1]
            
            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
    
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every*size)
            print_loss_total = 0
            print('%s: %s (%d %d%%) %.4f' % (start1,timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
    
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / (plot_every*size)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    showPlot(plot_losses)




def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    f=open("output/output[80-20]_1.txt","w")
    for pair in test:
        #pair = random.choice(test)
        f.write('>'+pair[0]+'\n')
        f.write('='+pair[1]+'\n')
        #print('>', pair[0])
        #print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        f.write('<'+output_sentence+'\n')
        #print('<', output_sentence)
        #print('')
    while True:
        s=input("Enter source text:")
        output_words, attentions = evaluate(encoder, decoder, s)
        output_sentence = ' '.join(output_words)
        print(output_sentence)
        c=input("Do you want to continue y or n:")
        if c == 'n' or c=='N':
            break

hidden_size = 500
encoder1 = AttnEncoderRNN(input_lang.n_words, hidden_size,n_layers=1, dropout_p=0.2)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.6)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainEpochs(encoder1, attn_decoder1, 25, print_every=1, plot_every=1,learning_rate=0.01)


print(encoder1,"\n",attn_decoder1)
torch.save(encoder1.state_dict(), "encoder.pt")
torch.save(attn_decoder1.state_dict(), "decoder.pt")
#encoder1.state_dict("encoder.pt")
#attn_decoder1.state_dict("decoder.pt")
evaluateRandomly(encoder1, attn_decoder1)


'''output_words, attentions = evaluate(
    encoder1, attn_decoder1, "यह लक्षण बिना किसी उपचार के समाप्त हो जाते हैं।")
plt.matshow(attentions.numpy())'''



def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


#evaluateAndShowAttention("डी.टी.पी. टीकों की 3 खुराक बच्चों को देकर।")
#evaluateAndShowAttention("यह लक्षण बिना किसी उपचार के समाप्त हो जाते हैं।")
#evaluateAndShowAttention("दो से तीन मिनट का समय लगता है।")

evaluateAndShowAttention("चुइंग गम चबाने से लार बनती है।")

#evaluateAndShowAttention("फलों और सब्जियों में ऐसा ही अनुपात रहता है।")
#evaluateAndShowAttention("चुइंग गम चबाने से लार बनती है।")
evaluateAndShowAttention("लड़का ने लड़की को मारा।")
