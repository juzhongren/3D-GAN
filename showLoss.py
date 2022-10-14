import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

#查看训练的loss
#训练集的loss
f=open('loss_history/train_loss_history.txt','rb')
train_loss = pickle.load(f)
f.close()
#训练集判别器的loss
f=open('loss_history/train_loss_D_history.txt','rb')
train_D_loss = pickle.load(f)
f.close()
#验证集loss
f=open('loss_history/val_loss_history.txt','rb')
val_loss = pickle.load(f)
f.close()
# 验证集判别器的loss
f=open('loss_history/val_loss_D_history.txt','rb')
val_D_loss = pickle.load(f)
f.close()

# 参数名：figsize
# 类型： tuple of integers 整数元组, optional可选, default: None，默认没有
# 备注：宽度，高度英寸。如果没有提供，默认为rc figure.figsize。
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(train_loss[:],label="G")
plt.plot(train_D_loss[:],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
#单独看
#loss损失
plt.figure(figsize=(10,5))
plt.title("Generator Loss During Training")
plt.plot(train_loss[:],label="G")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
#D_loss
plt.figure(figsize=(10,5))
plt.title("Discriminator Loss During Training")
plt.plot(train_D_loss[:],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#验证集损失
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Validation")
plt.plot(val_loss[:],label="G")
plt.plot(val_D_loss[:],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#验证集损失loss
plt.figure(figsize=(10,5))
plt.title("Generator Loss During Validation")
plt.plot(val_loss[:],label="G")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#验证集损失D_loss
plt.figure(figsize=(10,5))
plt.title("Discriminator Loss During Validation")
plt.plot(val_D_loss[:],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()