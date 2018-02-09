


D = read.csv('save/rnn/temperature.csv')
# names(D.agg) <- c("Temperature","Valid Generation Rate","Average SMILES Length")

library(ggplot2)
library(reshape2)

temp.valid = ggplot(D[,c(1,2)],aes(x=Temperature,y=Valid)) +
	geom_line() +
	scale_x_continuous(breaks = seq(0, 1, by = 0.05)) +
	scale_y_continuous(lim=c(0.0,1.0),breaks = seq(0.0, 1, by = 0.05)) +
	labs(x="Temperature",y="Percent Valid") +
	theme_bw()

ggsave('temp_valid.png', plot=temp.valid, dpi=300, width=8, height=4)

temp.len = ggplot(D[,c(1,5)],aes(x=Temperature,y=Length)) +
	geom_line() +
	scale_x_continuous(breaks = seq(0, 1, by = 0.05)) +
	labs(x="Temperature",y="SMILES Length") +
	theme_bw()

ggsave('temp_length.png', plot=temp.len, dpi=300, width=8, height=4)


D = read.csv('save/rnn/history_pre-finetuning.csv')
names(D) <- c("Training Accuracy","Training Loss","Validation Accuracy","Validation Loss","Learning Rate")
D = cbind(Epoch=1:nrow(D), D)

D.trn = D[,c(1,2,4)]
names(D.trn) = c("Epoch", "Training","Validation")

D.melt = melt(D.trn,id="Epoch")
names(D.melt)[3] = "Value"
hist.acc = ggplot(D.melt, aes(x=Epoch,y=Value,group=rev(variable),color=variable)) +
	geom_line() +
	scale_x_continuous(breaks = seq(0, 400, by = 50)) +
	scale_y_continuous(lim=c(0,1),breaks = seq(0, 1, by = 0.1)) +
    scale_color_manual(values = c("black", "orange")) +
	labs(x="Epoch",y="Accuracy",legend="") +
	theme_bw() +
	theme(legend.title=element_blank(),legend.justification=c(1,0), legend.position=c(0.98,0.02))

ggsave("history_acc.png",plot=hist.acc,dpi=300,width=8,height=4)


D.val = D[,c(1,3,5)]
names(D.val) = c("Epoch", "Training","Validation")
D.melt = melt(D.val,id="Epoch")
names(D.melt)[3] = "Value"
hist.loss = ggplot(D.melt, aes(x=Epoch,y=Value,group=rev(variable),color=variable)) +
	geom_line() +
	scale_x_continuous(breaks = seq(0, 400, by = 50)) +
	scale_y_continuous(breaks = seq(0, 3, by = 0.2)) +
    scale_color_manual(values = c("black", "orange")) +
	labs(x="Epoch",y="Categorical Cross-Entropy",legend="") +
	theme_bw() +
	theme(legend.title=element_blank(),legend.justification=c(1,1), legend.position=c(0.98,0.98))

ggsave("history_loss.png",plot=hist.loss,dpi=300,width=8,height=4)
