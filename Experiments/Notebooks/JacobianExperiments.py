import jacobian, numpy as np, torch

x = torch.randn(10,32,5).cuda()
x.requires_grad = True
y = torch.nn.LSTM(input_size=5, hidden_size=10, num_layers=2).cuda()
outputs, hidden = y(x)
op = outputs[0]
mse = torch.nn.MSELoss()
loss = mse(op, torch.ones_like(op).cuda()) 

optimizer = torch.optim.Adam(y.parameters(), lr=1e-1)
optimizer.zero_grad()

jacobian_regularizer = jacobian.JacobianReg()
jacregloss = jacobian_regularizer(x, op)

total_loss = loss+jacregloss
total_loss.backward()
optimizer.step()

#######################
#######################

import jacobian, numpy as np, torch

x = torch.randn(32,5).cuda()
x.requires_grad = True
# y = torch.nn.LSTM(input_size=5, hidden_size=10, num_layers=2).cuda()
y = torch.nn.Linear(5,7).cuda()
# outputs, hidden = y(x)
# op = outputs[0]
op = y(x)
mse = torch.nn.MSELoss()
loss = mse(op, torch.ones_like(op).cuda()) 

optimizer = torch.optim.Adam(y.parameters(), lr=1e-1)
optimizer.zero_grad()

jacobian_regularizer = jacobian.JacobianReg()
jacregloss = jacobian_regularizer(x, op)

total_loss = loss+jacregloss
total_loss.backward()
optimizer.step()

#######################
#######################

import numpy as np, torch

x = torch.randn(10,32,5).cuda()
x.requires_grad = True
y = torch.nn.LSTM(input_size=5, hidden_size=10, num_layers=2).cuda()
outputs, hidden = y(x)
op = outputs

optimizer = torch.optim.Adam(y.parameters(), lr=1e-1)
optimizer.zero_grad()

gradients = torch.autograd.grad(outputs=op, inputs=x, grad_outputs=torch.ones_like(op).cuda(), create_graph=True, retain_graph=True)


#######################
#######################

import numpy as np, torch

x = torch.randn(4,3,21).cuda()
x.requires_grad = True
y = torch.nn.Linear(21, 4).cuda()
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=4, nhead=4, dropout=0., dim_feedforward=24).cuda()
transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6).cuda()
# decoder_layer = torch.nn.TransformerDecoderLayer(d_model=4, nhead=4).cuda()
# transformer_decoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6).cuda()
# y = torch.nn.LSTM(input_size=5, hidden_size=10, num_layers=2).cuda()
z = y(x)
op = transformer_encoder(z)

optimizer = torch.optim.Adam(transformer_encoder.parameters(), lr=1e-1)
optimizer.zero_grad()

# gradients = torch.autograd.grad(outputs=op, inputs=x, grad_outputs=torch.ones_like(op).cuda(), create_graph=True, retain_graph=True)[0]
gradients = torch.autograd.grad(outputs=op, inputs=z, grad_outputs=torch.ones_like(op).cuda(), create_graph=True, retain_graph=True)[0]

loss = op.mean()
grad_loss = torch.norm(gradients)
total_loss = loss+grad_loss
total_loss.backward()
optimizer.step()

#######################
#######################

# gradients = torch.autograd.grad(outputs=op[0], inputs=x[0], grad_outputs=torch.ones_like(op[0]).cuda(), create_graph=True, retain_graph=True)[0
g0 = torch.autograd.grad(outputs=op[0], inputs=z, grad_outputs=torch.ones_like(op[0]).cuda(), create_graph=True, retain_graph=True)[0]
g1 = torch.autograd.grad(outputs=op[1], inputs=z, grad_outputs=torch.ones_like(op[0]).cuda(), create_graph=True, retain_graph=True)[0]
g2 = torch.autograd.grad(outputs=op[2], inputs=z, grad_outputs=torch.ones_like(op[0]).cuda(), create_graph=True, retain_graph=True)[0]
g3 = torch.autograd.grad(outputs=op[3], inputs=z, grad_outputs=torch.ones_like(op[0]).cuda(), create_graph=True, retain_graph=True)[0]

go0 = torch.autograd.grad(outputs=op, inputs=z[0], grad_outputs=torch.ones_like(op).cuda(), create_graph=True, retain_graph=True, allow_unused=True)[0]
go1 = torch.autograd.grad(outputs=op, inputs=z[1], grad_outputs=torch.ones_like(op).cuda(), create_graph=True, retain_graph=True)[0]
go2 = torch.autograd.grad(outputs=op, inputs=z[2], grad_outputs=torch.ones_like(op).cuda(), create_graph=True, retain_graph=True)[0]
go3 = torch.autograd.grad(outputs=op, inputs=z[3], grad_outputs=torch.ones_like(op).cuda(), create_graph=True, retain_graph=True)[0]
gij = torch.autograd.grad(outputs=op[:,i], inputs=z[:,j], grad_outputs=torch.ones_like(op[:,0]).cuda(), create_graph=True, retain_graph=True)[0]

# 
for k in range(3,4):
    for j in range(4):
        print("#######################", k, j)    
        g = torch.autograd.grad(outputs=op[k], inputs=z[j], grad_outputs=torch.ones_like(op[0]).cuda(), create_graph=True, retain_graph=True)[0]
        # print("#######################", k, j)        
        print(g)
        print("#######################")
        print("#######################")

##############################################
##############################################
        
import numpy as np, torch

x = torch.randn(4,3,21).cuda()
# q = 10*torch.randn(4,1,4).cuda()
x.requires_grad = True
y = torch.nn.Linear(21, 4).cuda()
transformer = torch.nn.Transformer(d_model=4, nhead=4, dropout=0., batch_first=False).cuda()
z = y(x)
tgt = torch.zeros_like(z[1]).cuda()
# tgt = 0.01 * torch.ones_like(z).cuda()
op = transformer(z, tgt)

optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
optimizer.zero_grad()
loss = ((op-q)**2).mean()
total_loss = loss
total_loss.backward()
optimizer.step()

z = y(x)
tgt = torch.zeros_like(z).cuda()
op = transformer(z, tgt)

# grad_loss = torch.norm(gradients)
total_loss = loss
total_loss.backward()
optimizer.step()

##############################################
##############################################

