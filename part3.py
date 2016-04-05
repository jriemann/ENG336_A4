"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('shakespeare_train.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 250 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

def sample_with_temp(h, seed_ix, n, temp):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  assert(temp>0)
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y/temp) / np.sum(np.exp(y/temp)) # This is where we apply temperature.
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

def complete_string(h, incomplete, max_len=100):
    """
    Use the rnn h to complete the string 'incomplete', up to max_len characters.
    """
    print("Completing string: " + str(incomplete))
    if len(incomplete) < 1:
        last = char_to_ix[" "]
    else:
        last = char_to_ix[incomplete[-1]]

    #x = np.zeros((vocab_size, 1))
    #x[last] = 1
    #ixes = []
    #hidden_activity = f(last) ???
    while len(incomplete) < max_len and (incomplete[-1] not in [" ", "\n"]):
        
        # generate a prediction using h.
        new_char = sample(h, last, 1)
        last = new_char
        incomplete = incomplete + ix_to_char[new_char[0]]
        

    return incomplete


PART_1 = False
PART_2 = True

if PART_1:
    temps = [0.1, 0.5, 0.9, 1.0, 5.0, 10.0]
else:
    temps = [1.0]

n, p = 0, 0

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
a = np.load(open("char-rnn-snapshot.npz"))
Wxh = a["Wxh"] 
Whh = a["Whh"]
Why = a["Why"]
bh = a["bh"]
by = a["by"]
mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
mbh, mby = a["mbh"], a["mby"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a["vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()

colon_ix = char_to_ix[':']
colon_x  = np.zeros((vocab_size, 1))
colon_x[colon_ix] = 1

newln_ix = char_to_ix['\n']
colon_x  = np.zeros((vocab_size, 1))
colon_x[colon_ix] = 1

colon_tanh = np.tanh(Wxh[:, colon_ix])
tanh_newln = Why[newln_ix, :]

colon_newln_vals = np.multiply(colon_tanh, tanh_newln)
colon_newln_wts = colon_newln_vals.argsort()[::-1]

i = 0

for wt in colon_newln_wts:
    print 'Path {} : char {} -> hidden {} -> output {}\n'.format(i, colon_ix, wt, newln_ix)
    tanh_val = np.tanh(Wxh[wt, colon_ix])
    outp_val = tanh_val * Why[newln_ix, wt]

    print 'Wxh[{}, {}]    = {:.4f}'.format(wt, colon_ix, Wxh[wt, colon_ix])
    print 'tanh( {:.4f} ) = {:.4f}'.format(Wxh[wt, colon_ix], tanh_val)
    print 'Why[{}, {}]    = {:.4f}'.format(newln_ix, wt, Why[newln_ix, wt])
    print 'y("\\n")        = {:.4f}\n\n'.format(outp_val)


    i += 1

