'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch
from typing import Optional, Union, Tuple, Type, Set

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):


    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding   

        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers. 

        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']

        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}

        ## what I am doing here:
        ## initialize embeddings and the rnn layer  

        ## initialize recurrent network architecture
        if self.cell_type.lower() == 'lstm':
            self.rnn = torch.nn.LSTM(input_size = self.word_embedding_size, 
                                     hidden_size = self.hidden_state_size, 
                                     num_layers = self.num_hidden_layers,
                                     bidirectional = True,
                                     dropout = self.dropout) 
        elif self.cell_type.lower() == 'gru':
            self.rnn = torch.nn.GRU(input_size = self.word_embedding_size,
                                    hidden_size = self.hidden_state_size,
                                    num_layers = self.num_hidden_layers,
                                    dropout = self.dropout,
                                    bidirectional = True)
        elif self.cell_type.lower() == "rnn":
            self.rnn = torch.nn.RNN(input_size = self.word_embedding_size,
                                    hidden_size = self.hidden_state_size,
                                    num_layers = self.num_hidden_layers,
                                    dropout = self.dropout,
                                    bidirectional = True)
        ## initialize a word embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings = self.source_vocab_size,
                                            embedding_dim = self.word_embedding_size,
                                            padding_idx = self.pad_id)


    def forward_pass(
            self,
            F: torch.LongTensor,
            F_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states

        ## what I am doing here:
        ## the two methods get_all_rnn_inputs and get_all_hidden_states are grouped in forward_pass
        ## which takes in words, returns hidden states 
        embedded = self.get_all_rnn_inputs(F = F)
        
        ## h is passed for attention
        h = self.get_all_hidden_states(x = embedded,
                                          F_lens = F_lens,
                                          h_pad = h_pad)
        #print(h.shape) 
        ## should be (S, M, 2 * self.hidden_state_size)
        ## s = is the number of source time steps
        ## M = batch dimension
        ## 2 is for forward and backward 
        return h

    def get_all_rnn_inputs(self, F: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   x (output) is shape (S, M, I)
        
        ## what I am doing here:
        ## accept a batch of source sequences and lengths S and output word embeddings for the sequences
        x = self.embedding(F)

        #print(x.shape)
        ## should be (S, M, I)
        ## s = is the number of source time steps
        ## M = batch dimension
        ## I = size of the per-word input vector
        return x

    def get_all_hidden_states(
            self, 
            x: torch.FloatTensor,
            F_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:
        # Recall:
        #   x is of shape (S, M, I)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #   h (output) is of shape (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence

        ## what I am doing here:
        ## call RNN on the inputs and and produce output representations for each of the words
        ## converts the word embeddings into hidden states for the last layer of the RNN

        ## 1. pad a packed batch of variable length sequences.                                        
        xPadded =  torch.nn.utils.rnn.pack_padded_sequence(input = x, # x = padded batch of variable length sequences
                                                lengths = F_lens.cpu(),  
                                                enforce_sorted = False) # otherwise I'll get an error as lengths isn't sorted
        
        ## 2. packs a tensor containing padded sequences of variable length.
        packed, hidden = self.rnn(xPadded)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence = packed,
                                                   padding_value = h_pad) # hpad = values for padded elements
        # print(h.shape)
        ## should be (S, M, 2 * self.hidden_state_size)
        ## s = is the number of source time steps
        ## M = batch dimension
        ## 2 is for forward and backward  

        return h


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}

        ## initialize a word embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings = self.target_vocab_size,
                                            embedding_dim = self.word_embedding_size,
                                            padding_idx = self.pad_id)
                                        
        ## initialize a recurrent cell
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size = self.word_embedding_size,
                                          hidden_size = self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size = self.word_embedding_size,
                                          hidden_size = self.hidden_state_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size = self.word_embedding_size,
                                          hidden_size = self.hidden_state_size)

        ## initialize a feed-forward layer to convert the hidden state to logits
        self.ff  = torch.nn.Linear(in_features = self.hidden_state_size,
                                   out_features = self.target_vocab_size)

    def forward_pass(
        self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> Tuple[
                torch.FloatTensor, Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   logits_t (output) is of shape (M, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.

        if htilde_tm1 is None:
            #float tensor of shape ``(M, self.hidden_state_size)``
            if self.cell_type in ['rnn', 'gru']:
                htilde_tm1 = self.get_first_hidden_state(h, F_lens)
            else:
                htilde_tm1 = self.get_first_hidden_state(h, F_lens)
                # pair of float tensors corresponding to the previous hidden state and the previous cell state
                htilde_tm1 = (htilde_tm1, torch.zeros_like(htilde_tm1))
            

        xtilde_t = self.get_current_rnn_input(E_tm1 = E_tm1,
                                              htilde_tm1 = htilde_tm1,
                                              h = h,
                                              F_lens = F_lens).to(E_tm1.device)                                  
        htilde_t = self.get_current_hidden_state(xtilde_t = xtilde_t,
                                                 htilde_tm1 = htilde_tm1)
        
        if self.cell_type == 'lstm':
            logits_t = self.get_current_logits(htilde_t = htilde_t[0])
        elif self.cell_type in ['rnn', 'gru']:
            logits_t = self.get_current_logits(htilde_t = htilde_t)
        
        # print(logits_t.shape, htilde_t.shape)
        return logits_t, htilde_t


    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   htilde_tm1 (output) is of shape (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat

        ## what I am doing here:
        ## takes h from the output of the encoder and use that to initialize htilda_1 as input to the decoder
        ## since we want in each direction, we process each separately and then concatenate

        ## at the last state

        forward = h[F_lens - 1, torch.arange(F_lens.shape[0]), :self.hidden_state_size//2]
        ## at the first state
        backward = h[0, :, self.hidden_state_size//2:]
        ## concatenate the two
        htilde_tm1 =torch.cat([forward, backward], dim =1)
        #print(htilde_tm1.shape)
        ## should be (M, self.hidden_state_size)
        ## M = batch dimension
        return htilde_tm1
            
    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   xtilde_t (output) is of shape (M, Itilde)
        
        ## get embeddings for relevant words 

        xtilde_t = self.embedding(E_tm1).to(h.device)
        #print(xtilde_t.shape)
        ## should be (M, Itilde)
        return xtilde_t

    def get_current_hidden_state(
            self,
            xtilde_t: torch.FloatTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]]) -> Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]:
        # Recall:
        #   xtilde_t is of shape (M, Itilde)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1

        ## what I am doing here:
        ## take in output from previous encoder representation and the input embedding 
        ## calculate the decoder's new hidden state
        
        if self.cell_type in ['lstm']:
            # tuple of float tensors
            htilde_tm1 = (htilde_tm1[0][:, :self.hidden_state_size], 
                          htilde_tm1[1][:, :self.hidden_state_size])
        elif self.cell_type in ['rnn', 'gru']:
            ## if decoder doesn't use an LSTM cell type there is only one float tensor
            htilde_tm1 = htilde_tm1[:, :self.hidden_state_size]
        
        # tuple (pair of float tensors) if lstm, and float tensor if other 
        htilde_t = self.cell(xtilde_t, htilde_tm1)

        return htilde_t

    def get_current_logits(
            self,
            htilde_t: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (M, V)

        ## what I am doing here:
        ## take hidden representations to get probability distribution over all vocabulary words
        ## return the un-normalized distribution over the next target word
        ## which will then be fed to the softmax function

        logits_t = self.ff(htilde_t)
        # print(logits_t.shape)
        ## should be (M, self.target_vocab_size)
        ## M = batch dimension
        ## self.target_vocab_size
        
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        
        ## initialize a word embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings = self.target_vocab_size,
                                            embedding_dim = self.word_embedding_size,
                                            padding_idx = self.pad_id)
                                        
        ## initialize a recurrent cell
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size = self.word_embedding_size + self.hidden_state_size,
                                          hidden_size = self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size = self.word_embedding_size + self.hidden_state_size,
                                          hidden_size = self.hidden_state_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size = self.word_embedding_size + self.hidden_state_size,
                                          hidden_size = self.hidden_state_size)

        ## initialize a feed-forward layer to convert the hidden state to logits
        self.ff  = torch.nn.Linear(in_features = self.hidden_state_size,
                                   out_features = self.target_vocab_size)


    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:

        # Hint: For this time, the hidden states should be initialized to zeros.

        ## what I am doing here:
        ## get the initial decoder hidden state, prior to the first input
        device = h.device
        S = h[0]
        htilde_0 = torch.zeros_like(S, device = device)

        #print(htilde_0.shape)
        ## should be (M, self.hidden_state_size)
        ## M = batch dimension
        ## self.hidden_state_size
        return htilde_0

    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: Use attend() for c_t
        
        if self.cell_type not in ['rnn', 'gru']:
            htilde_tm1 = htilde_tm1[0]
        ## get the embedding
        embedded = self.embedding(E_tm1)
        ## call attend to get c_t
        c_t = self.attend(htilde_tm1, h, F_lens)
        ## concatenate the c_t and the embedding 
        xtilde_t =  torch.cat([embedded, c_t], dim =1) 
        # print(xtilde_t.shape)
        ## should be (M, Itilde)
        return xtilde_t

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        alpha_t = self.get_attention_weights(htilde_t = htilde_t, 
                                        h = h,
                                        F_lens = F_lens).unsqueeze(2)
        c_t = (alpha_t*h).sum(dim = 0)

        return c_t

    def get_attention_weights(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_attention_scores()
        # alpha_t (output) is of shape (S, M)
        e_t = self.get_attention_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens.to(h.device)  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_attention_scores(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity
        # import pdb; pdb.set_trace()

        ## what I am doing here:
        ## cosine similarity between hidden state for previous time step and encoded representaitions
        ## softmax already implemented for us 

        if self.cell_type not in ['rnn','gru']:
            htilde_t = htilde_t.unsqueeze(0)
        else:
            
            htilde_t = htilde_t[0].unsqueeze(0)
        
        e_t = torch.nn.functional.cosine_similarity(htilde_t, h, dim = 2)

        return e_t
        
class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        self.W = torch.nn.Linear(in_features = self.hidden_state_size,
                                 out_features = self.hidden_state_size,
                                 bias = False)
        self.Wtilde = torch.nn.Linear(in_features = self.hidden_state_size,
                                 out_features = self.hidden_state_size,
                                 bias = False)
        self.Q = torch.nn.Linear(in_features = self.hidden_state_size,
                                 out_features = self.hidden_state_size,
                                 bias = False)
        

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point

        ## what I am doing here:
        ## transform each of the hidden representations for each attention head
        ## running several different attention processes using different transformed representations

        #import pdb; pdb.set_trace()
        ## linear transformation applied to hidden state and from previous step
        #if self.cell_type == 'lstm':
        #    htilde_tn = self.Wtilde(htilde_t[0])
        #if self.cell_type != 'lstm':
        htilde_tn = self.Wtilde(htilde_t)
        
        htilde_tn = htilde_tn.view(htilde_tn.shape[0]*self.heads, self.hidden_state_size//self.heads)

        hs_n = self.W(h)
        hs_n = hs_n.view(hs_n.shape[0], hs_n.shape[1]*self.heads, self.hidden_state_size//self.heads)

        F_lens_il = F_lens.repeat_interleave(repeats = self.heads)
        ## use attend method of the super() class
        c_t_concat = super().attend(htilde_tn, hs_n, F_lens_il)
        ## concatenate 
        c_t_concat = c_t_concat.view(htilde_t.shape[0], htilde_t.shape[1])
        ## return Q matrix times output of attention, c_t_concat
        c_t = self.Q(c_t_concat)
        return c_t

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(
            self,
            encoder_class: Type[EncoderBase],
            decoder_class: Type[DecoderBase]):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it

        ## initialize an encoder and a decoder 
        self.encoder = encoder_class(source_vocab_size = self.source_vocab_size, 
                                    pad_id = self.source_pad_id, 
                                    num_hidden_layers = self.encoder_num_hidden_layers,
                                    word_embedding_size = self.word_embedding_size,
                                    dropout = self.encoder_dropout,
                                    hidden_state_size = self.encoder_hidden_size,
                                    cell_type = self.cell_type)
        self.encoder.init_submodules()
        
        self.decoder = decoder_class(target_vocab_size = self.target_vocab_size,
                                    word_embedding_size = self.word_embedding_size,
                                    pad_id = self.target_eos,
                                    heads = self.heads,
                                    hidden_state_size = self.encoder_hidden_size * 2,
                                    cell_type = self.cell_type)
        self.decoder.init_submodules()


    def get_logits_for_teacher_forcing(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor,
            E: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   E is of shape (T, M)
        #   logits (output) is of shape (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
            ## we don't have one for the last state
        

        ## what I am doing here:
        ## un-normed distributions over next tokens via teacher forcing 
        ## feed predicted word as input to next hidden state
        ## instead of using predicted input during training, use the true input E

        htilde_tm1 = None

        logits = []
        for time in range(E.shape[0]-1):
            true_input = E[time]
            curr_logits, htilde_tm1 = self.decoder.forward_pass(true_input, 
                                                                htilde_tm1, 
                                                                h, 
                                                                F_lens)
            logits += [curr_logits]
        
        logits = torch.stack(logits[:], 0)
        # print(logits.shape)
        # should be (T - 1, M, self.target_vocab_size)
        return logits

    def update_beam(
            self,
            htilde_t: torch.FloatTensor,
            b_tm1_1: torch.LongTensor,
            logpb_tm1: torch.FloatTensor,
            logpy_t: torch.FloatTensor) -> Tuple[
                torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (M, K)
        #   b_tm1_1 is of shape (t, M, K)
        #   b_t_0 (first output) is of shape (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of shape (t + 1, M, K)
        #   logpb_t (third output) is of shape (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]

        V = logpy_t.shape[-1]
        

        log_probs = logpb_tm1.unsqueeze(-1) + logpy_t

        paths = log_probs.view((log_probs.shape[0], -1))

        logpb_t, indexes = paths.topk(self.beam_width, -1, 
                                       largest = True, 
                                       sorted = True)
        # chosen path
        chosen_path = torch.div(indexes, V)
        ## remainder
        indexes = torch.remainder(indexes, V)
        ## concat new words
        path_b_tm1_1 = b_tm1_1.gather(2, chosen_path.type(torch.int64).unsqueeze(0).expand_as(b_tm1_1))

        if self.cell_type in ['lstm']:
            first = htilde_t[0].gather(1, chosen_path.type(torch.int64).unsqueeze(-1).expand_as(htilde_t[0]))
            second = htilde_t[1].gather(1, chosen_path.type(torch.int64).unsqueeze(-1).expand_as(htilde_t[1]))
            b_t_0 = (first, second)
        else: 
            b_t_0 = htilde_t.gather(1, chosen_path.type(torch.int64).unsqueeze(-1).expand_as(htilde_t))
            
        indexes = indexes.unsqueeze(0) 

        b_t_1 = torch.cat([path_b_tm1_1, indexes], dim = 0)
        #print(b_t_0.shape, b_t_1.shape, logpb_t.shape)
        return b_t_0, b_t_1, logpb_t