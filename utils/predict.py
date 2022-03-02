import torch
from torch import nn
import torch.nn.functional as func
import numpy as np
import torch.nn.functional as F

class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # Dictionnaires des caractères 
        self.chars = tokens
        self.int_to_char = dict(enumerate(self.chars))
        self.char_to_int = {ch: ii for ii, ch in self.int_to_char.items()}
        
        ## Définir le LSTM 
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## Définir la couche de dropout
        self.dropout = nn.Dropout(drop_prob)
        
        ## Définir la couche de sortie
        self.fc = nn.Linear(n_hidden, len(self.chars))
      
    
    def forward(self, x, hidden):
                
        ## Recevoir la sortie et le nouvel état 'hidden' depuis le LSTM
        r_output, hidden = self.lstm(x, hidden)
        
        ## Passer par une couche de dropout
        out = self.dropout(r_output)
        
        # Empiler (stacking) les sorties des LSTM 
        # Ici contiguous() c'est pour faire le reshape du vecteur 
        out = out.contiguous().view(-1, self.n_hidden)
        
        ## Faire passer x dans la couche entièrement connecté
        out = self.fc(out)
        
        # Retourner la sortie finale et le hidden state final
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initialiser le hidden state '''
        # Créations de 2 tensors de dimensions n_layers x batch_size x n_hidden,
        # initialiser à 0 pour le hidden state et cell state (cellule état) du LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden


def predict(net, char, h=None, top_k=None):
        
        x = np.array([[net.char_to_int[char]]])
        x = one_hot(x, len(net.chars))
        inputs = torch.from_numpy(x)
        
        
        h = tuple([each.data for each in h])
        
        out, h = net(inputs, h)

        # Calculer les probabilités du prochain caractère
        p = F.softmax(out, dim=1).data
       
        
        # Garder les caractères avec les plus grandes probabiités
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # Choisir le prochain caractère à l'aide d'une fonction aléatoire
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # Retourner le code du caractère et le convertir en caractère  
        return net.int_to_char[char], h

def sample(net, size, prime='The', top_k=None):
    
    net.eval() # mode d'évaluation
    
    # D'abord, traiter le premier 
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Passer le caractère précédent pour avoir le prochaine
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print(sample(net, 300, prime='السلام', top_k=5))