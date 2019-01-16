
loss_batch = [[] for j in range(32)]
for j in range(32):
    loss_batch[j] = j+1
loss_all = sum(loss_batch)/32
x = 33*16/32
print(loss_all,x)