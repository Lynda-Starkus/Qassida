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



# Ouvrir le fichier texte lire les données sous forme de 'texte'

with open('poems_of_Darwish.txt', 'r', encoding="utf-8") as f:
  poem = f.read()


chars = tuple(set(poem))
int_to_char = dict(enumerate(chars))
char_to_int = {ch: ii for ii, ch in int_to_char.items()}

encoded = np.array([char_to_int[ch] for ch in poem])

def one_hot(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot


def get_batches(arr, batch_size, seq_length):

    batch_size_total = batch_size * seq_length

    # Nombre total de batches que nous pouvons avoir
    n_batches = len(arr)//batch_size_total
    
    # Garder que les batches complets 
    arr = arr[:n_batches * batch_size_total]

    # Reshape en lignes de batch_size 
    arr = arr.reshape((batch_size, -1))
    
    # Itérer sur chaque séquence de caractères
    for n in range(0, arr.shape[1], seq_length):

        x = arr[:, n:n+seq_length]
        
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

batches = get_batches(encoded, 8, 50)
x, y = next(batches)

train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('L\'entrainement se fera sur GPU')
else: 
    print('Aucun GPU trouvé ')


def train(net, data, epochs=5, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Entrainer le modèle
    
        Arguments
        ---------
        
        net: Le réseau charRNN
        data: Texte du dataset 
        epochs: Nombre d'epochs pour entrainer 
        batch_size: Nombre de mini-séquences par mini-batch (taille du batch)

        seq_length: Nombre de caractères par mini-batch
        lr: learning rate (taux d'apprentissage)
        clip: gradient clipping
        val_frac: Fraction de données à garder pour l'étape de validation
        print_every: Nombre d'étapes à compter jusqu'au prochain affichage du loss
    
    '''
    train_loss = 0.0
    valid_loss = 0.0
    valid_loss_min = np.Inf 


    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Création des données de less et de validation
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialiser l'état hidden 
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            x = one_hot(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            # initialiser les gradients cumulés 
            net.zero_grad()
            
            # Recevoir la sortie 
            output, h = net(inputs, h)
            
            # Calculer la perte (loss) et effectuer une backpropagation
            loss = train_loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            # `clip_grad_norm` permet de controler l'explosion du gradient dans RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # statistiques du loss
            if counter % print_every == 0:
                
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    
                    x = one_hot(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = valid_loss =  criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train() # Remettre en état d'entrainement après la phase de validation
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Etape: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

# Définir le network (réseau)
n_hidden=128
n_layers=3

net = CharRNN(chars, n_hidden, n_layers)
print(net)

batch_size = 32
seq_length = 100
n_epochs = 100

# Entrainer le modèle 
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

model_name = 'Darwish_model.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(model_name, f)