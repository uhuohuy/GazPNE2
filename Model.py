import torch
import torch.nn as nn
import torch.nn.functional as F

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim

class SelfAttentionLSTM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        # encoder_outputs = [batch size, sent len, hid dim]
        energy = self.projection(encoder_outputs)
        # energy = [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # weights = [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # outputs = [batch size, hid dim]
        return outputs, weights
    
class SelfAttention(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(filters, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        # encoder_outputs = [batch size, sent len, filters]
        energy = self.projection(encoder_outputs)
        # energy = [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # weights = [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1))
        # outputs = [batch size, sent len, filters]
        return outputs

# only use 3-gram filter and one 
class C_LSTM(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_size, hidden_dim ,output_dim, flex_feat_len, dropout):
        super().__init__()
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size,self.embedding_dim+flex_feat_len)) 
        self.lstm = nn.LSTM(n_filters, hidden_dim, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hc): 
        #x = [sent len, batch size]
        #x = x.permute(1, 0)   
        #x = [batch size, sent len]
        embedded = self.embedding(x)
        if hc:
            embedded = torch.cat((embedded,hc),2) 
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim]
        #hidden = [1, batch size, hid dim]
        hidden = hidden.squeeze(0)
        #hidden = [batch size, hid dim]
        return self.fc(self.dropout(hidden))
    def predict(self,embedded):
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim]
        #hidden = [1, batch size, hid dim]
        hidden = hidden.squeeze(0)
        return self.fc(self.dropout(hidden))

# use only 3-gram filter and bidirection
class C_LSTMAttention(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_size, bidirectional, hidden_dim , output_dim, flex_feat_len, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size,self.embedding_dim+flex_feat_len)) 
        self.lstm = nn.LSTM(n_filters, hidden_dim, bidirectional=bidirectional, dropout=dropout)
        self.attention = SelfAttentionLSTM(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x,hc): 
        #x = [sent len, batch size]
        #x = x.permute(1, 0)   
        #x = [batch size, sent len]
        embedded = self.embedding(x)     
        #embedded = [batch size, sent len, emb dim]
        embedded = torch.cat((embedded,hc),2)
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim*num directional]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # output = [sent len, batch size, hid dim]
        ouput = output.permute(1, 0, 2)
        # ouput = [batch size, sent len, hid dim]
        new_embed, weights = self.attention(ouput)
        # new_embed = [batch size, hid dim]
        # weights = [batch size, sent len]
        new_embed = self.dropout(new_embed)
        return self.fc(new_embed)

    def predict(self,embedded):
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = F.relu(self.conv(embedded)).squeeze(3)
        #conved = [batch size, n_filters, sent len - filter_size]
        x = conved.permute(2, 0, 1)
        #x = [sent len- filter_size, batch size, n_filters]
        output, (hidden, cell) = self.lstm(x)
        #output = [sent len - filter_size, batch size, hid_dim*num directional]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # output = [sent len, batch size, hid dim]
        ouput = output.permute(1, 0, 2)
        # ouput = [batch size, sent len, hid dim]
        new_embed, weights = self.attention(ouput)
        # new_embed = [batch size, hid dim]
        # weights = [batch size, sent len]
        new_embed = self.dropout(new_embed)
        return self.fc(new_embed)
    
class AttentionCNN(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_sizes, output_dim, flex_feat_len, dropout=0.5):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,self.embedding_dim+flex_feat_len)) for fs in filter_sizes])
        self.attention = SelfAttention(n_filters)
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,hc):
        #x = [sent len, batch size]
        #x = x.permute(1, 0)   
        #x = [batch size, sent len]
        embedded = self.embedding(x)
        embedded = torch.cat((embedded,hc),2)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        conved_att = [(self.attention(conv.permute(0, 2, 1))).permute(0, 2, 1) for conv in conved]
        #conved_att = [batch size, n_filters, sent len - filter sizes[i]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_att]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
    def predict(self,embedded):
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        conved_att = [(self.attention(conv.permute(0, 2, 1))).permute(0, 2, 1) for conv in conved]
        #conved_att = [batch size, n_filters, sent len - filter sizes[i]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_att]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
    


class CNN(nn.Module):
    def __init__(self, weights_matrix, n_filters, filter_sizes, output_dim,flex_feat_len,
                 dropout=0.5):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.embedding,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        #
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fsd, self.embedding_dim+flex_feat_len)) 
                                    for fsd in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def core(self, embedded):
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)

        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)

    def forward(self, text,hc):
                
        #text = [batch size, sent len]
        embedded = self.embedding(text)
        embedded = torch.cat((embedded,hc),2)
        output = self.core(embedded)        
        return output


class BiLSTM(nn.Module):

    def __init__(self, weights_matrix, tagset_size, hidden_dim,num_layers, flex_feat_len, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.num_layers = num_layers
        self.word_embeds,self.vocab_size,self.embedding_dim = create_emb_layer(weights_matrix, True)
        
        self.lstm = nn.LSTM(self.embedding_dim+flex_feat_len, hidden_dim // 2,
                            self.num_layers, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        self.relu = nn.Tanh()
        #self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1)
        self.prediction = nn.Softmax(dim=1)

    def init_hidden(self,barch_size=100):
        return (torch.randn(2*self.num_layers, barch_size, self.hidden_dim // 2),
                torch.randn(2*self.num_layers, barch_size, self.hidden_dim // 2))

    def predict(self,embeds):
        barch_size=embeds.size()[0]
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(embeds)
        word_size=lstm_out.size()[1]
        last_cell_out = lstm_out.narrow(1,word_size-1,1)
        drop_out_feat = self.dropout(last_cell_out)
        lstm_feats = self.hidden2tag(drop_out_feat)
        score = self.prediction(lstm_feats.view(barch_size,self.tagset_size))
        return score

    def _get_lstm_features(self, sentence,hc):
        #self.hidden = self.init_hidden(barch_size), self.hidden

        embeds = self.word_embeds(sentence)

        embeds = torch.cat((embeds,hc),2)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(embeds)
        return lstm_out,final_hidden_state, final_cell_state

    def forward(self, sentence, hc):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        barch_size=sentence.size()[0]
        lstm_out, final_hidden_state, final_cell_state = self._get_lstm_features(sentence,hc)
        word_size=lstm_out.size()[1]
        last_cell_out = lstm_out.narrow(1,word_size-1,1)
        drop_out_feat = self.dropout(last_cell_out)
        #word_size=drop_out_feat.size()[1]
        #lstm_feats = self.hidden2tag(drop_out_feat.narrow(1,word_size-1,1))
        lstm_feats = self.hidden2tag(drop_out_feat)
        score = self.softmax(lstm_feats.view(barch_size,self.tagset_size))
        return score


