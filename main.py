import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pprint import pprint
import model
from model import EncoderDecoder
from data import data_loader
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# *Hyperparams

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default="/home/")
parser.add_argument('--data_path', type=str, default="/home/flickr8k/Images")
parser.add_argument('--model_path', type=str, default='/home/ckpt' , help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
parser.add_argument('--caption_path', type=str, default="/home/flickr8k/captions.txt")
parser.add_argument('--device', type=str, default="cpu")

# Model parameters
parser.add_argument('--embed_dim', type=int , default=300, help='dimension of word embedding vectors')
parser.add_argument('--attention_dim', type=int , default=512, help='dimension of attention linear layers')
parser.add_argument('--encoder_dim', type=int , default=2048, help='dimension of encoder resnet output')
parser.add_argument('--decoder_dim', type=int , default=512, help='dimension of decoder rnn')
parser.add_argument('--vocab_size', type=int , default=1, help='vocabulary size')
parser.add_argument('--drop_prob', type=float , default=0.3)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--print_every', type=int, default=10)
args = parser.parse_args()
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pprint(args)

dataset, train_loader, args.vocab_size = data_loader(args)

# initialize model, loss etc
model = EncoderDecoder(args).to(args.device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Early stopping
best_loss = np.inf
patience = 0

def save_model(model,num_epochs,args):
    model_state = {
        'num_epochs':num_epochs,
        'embed_size':args.embed_dim,
        'vocab_size':args.vocab_size,
        'attention_dim':args.attention_dim,
        'encoder_dim':args.encoder_dim,
        'decoder_dim':args.decoder_dim,
        'state_dict':model.state_dict()
    }
    torch.save(model_state,'attention_model_state.pth')

print("Start Training")
for epoch in range(1,args.epochs+1):   
    for idx, (image, captions) in enumerate(iter(train_loader)):
        model.train()
        image,captions = image.to(args.device),captions.to(args.device)
        # Zero the gradients.
        optimizer.zero_grad()
        # Feed forward
        outputs = model(image, captions) # (pred, alphas)
        # Calculate the batch loss.
        loss = criterion(outputs[0].view(-1, args.vocab_size), captions[:,1:].reshape(-1))
        # Backward pass.
        loss.backward()
        # Update the parameters in the optimizer.
        optimizer.step()
        
        if (idx+1)%args.print_every == 0:
            print("Evaluating")
            print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
            model.eval()
            with torch.no_grad():
                dataiter = iter(train_loader)
                img, caption = next(dataiter)
                all_actual = list()
                all_predicted = list()

                for i in range(caption.size()[0]):
                    actual = [dataset.vocab.itos[token] for token in caption[i].tolist() if token not in {0,1}] # except for <SOS> and <PAD>
                    features = model.encoder(img[i:i+1].to(args.device))
                    predicted, alphas = model.decoder.generate_caption(features,vocab=dataset.vocab, max_len=30)
                    
                    all_actual.append(actual);all_predicted.append(predicted)
                print("actual ",actual,"\t", "predicted ",predicted)    
                print('BLEU-1: %f' % corpus_bleu(all_actual, all_predicted, weights=(1.0, 0, 0, 0)))
                print('BLEU-2: %f' % corpus_bleu(all_actual, all_predicted, weights=(0.5, 0.5, 0, 0)))
                print('BLEU-3: %f' % corpus_bleu(all_actual, all_predicted, weights=(0.3, 0.3, 0.3, 0)))
                print('BLEU-4: %f' % corpus_bleu(all_actual, all_predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    
    if loss <= best_loss:
                patience_counter = 0
                best_epoch = epoch + 1
                best_loss = loss
                save_model(model,best_epoch,args)

    else:
        patience += 1
        if patience == (args.patience - 10):
            print('\nPatience counter {}/{}.'.format(
                patience, args.patience))
        elif patience == args.patience:
            print('\nEarly stopping... no improvement after {} Epochs.'.format(
                args.patience))
            break

    print("Epoch: {}\t Train Loss: {:.4f}".format(epoch+1, loss))
    loss = None
    