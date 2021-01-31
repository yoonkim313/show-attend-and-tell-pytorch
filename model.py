import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from nltk.translate.bleu_score import corpus_bleu

class EncoderCNN(nn.Module):
    ''' 
    Extracts features from image by using lower layer of CNN
    Total of L(512) d(196)-dimentional vectors show feature of a part of the image
    '''
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # Load imageNet pretrained resnet
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        # only cnn, excluding fc layers: extracts features by part
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)             #(bs, 7, 7, 2048)                  
        features = features.permute(0, 2, 3, 1)   
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features


class Attention(nn.Module):
    '''
    Badanau attention : soft attention weighted annotation vectors
    encoder_dim: 2048 (encoder output)
    decoder_dim: 512
    attention_dim: 512
    '''

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        
        self.A = nn.Linear(attention_dim,1)
        
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size, num_layers)
        
        return alpha, attention_weights
        


class DecoderRNN(nn.Module):
    '''
    encoder_dim: 2048 CNN encoded output dimension
    decoder_dim: 512
    attention_dim: 512
    embed_dim: 512 word embedding size
    '''
    def __init__(self, args):
        super().__init__()
        
        #save the model param
        self.args = args
        self.vocab_size = args.vocab_size
        self.attention_dim = args.attention_dim
        self.decoder_dim = args.decoder_dim
        
        self.embedding = nn.Embedding(args.vocab_size,args.embed_dim)
        self.attention = Attention(args.encoder_dim, args.decoder_dim, args.attention_dim)
        
        
        self.init_h = nn.Linear(args.encoder_dim, args.decoder_dim)  
        self.init_c = nn.Linear(args.encoder_dim, args.decoder_dim)  
        self.lstm_cell = nn.LSTMCell(args.embed_dim+args.encoder_dim,args.decoder_dim,bias=True)
        self.f_beta = nn.Linear(args.decoder_dim, args.encoder_dim)
        
        
        self.fcn = nn.Linear(args.decoder_dim, args.vocab_size)
        self.drop = nn.Dropout(args.drop_prob)
        
        
    
    def forward(self, features, captions):
        '''
        features: encoded image
        captions: label for a given image
        '''
        
        # embed the caption into word vector
        embeds = self.embedding(captions)
        # Initialize LSTM state
        
        h, c = self.init_hidden_state(features)  # features: (bs, 49, encoder_dim) -> (batch_size, decoder_dim)
        
        seq_length = len(captions[0])-1 # Exclude the last one
        batch_size = captions.size(0) # 128
        num_features = features.size(1) # 49
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.args.device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(self.args.device)
                
        for s in range(seq_length):
            alpha,context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        
        return preds, alphas

    # * Inference
    def generate_caption(self,features,max_len=30,vocab=None):

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        alphas = []
        
        #starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(self.args.device)
        embeds = self.embedding(word)
        
        captions = []
        
        for i in range(max_len):
            alpha,context = self.attention(features, h)
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            lstm_input = torch.cat((embeds.reshape(1,-1), context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
        
            
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        
        # vocab.stoi/ vocab.itos
        return [vocab.itos[idx] for idx in captions],alphas
    
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

class EncoderDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(args)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs # (pred, alphas)