library(magrittr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)
library(reshape2)
library(ggrepel)
library(scales)
library(zoo)

### Pretty Plots
scaleFUN <- function(x) sprintf("%.3f", x)
myTheme = theme(panel.background = element_rect(fill = NA, color = "black"),
                panel.grid.major = element_line(color = "#CCCCCC"),
                panel.grid.minor = element_line(color = "#EEEEEE"),
                strip.text = element_text(size=20),
                #legend.position = c(0.15,0.2),
                #legend.background = element_blank(),
                legend.box.background = element_rect(colour = "black"),
                panel.spacing = unit(2, "lines"),
                strip.background = element_blank(),
                axis.title.y = element_text(size=20),
                axis.title.x = element_text(size=20),
                axis.text.x = element_text(size=10, angle = 45, hjust = 1 ),
                axis.text.y = element_text(size=12),
                plot.title = element_text(hjust = 0.5, size=14)
)

####speed histogram
#othistdata = data.frame(read.csv("results/wassimplresults.csv"))
othistdata = data.frame(read.csv("results/wassimplresults_big.csv"))
#othistdata = data.frame(read.csv("results/ot_1d_results.csv"))

#speeddata = subset(speedhistdata, gradType!='torchdual')
#speeddata = subset(speeddata, gradType!='autograd')
#speeddata = subset(speedhistdata, gradType!='npdual')
#speeddata = subset(speeddata, gradType!='scipy')
#othistdata = subset(othistdata, n==100)
#othistdata = subset(othistdata, n==2)
othistdata = subset(othistdata, nbins==10)

dorder = c("2", "5", "10", "20", "50", "100")
othistdata$d <- factor(as.character(othistdata$d), levels = dorder)
#torder = c("demd", "sink_1d_bary", "lp_1d_bary", "cvx")
torder = c("demd", "pairwass")
othistdata$model <- factor(as.character(othistdata$model), levels = torder)

gg <- ggplot(othistdata, aes(x=d, y=time_per_epoch, fill=distType)) + 
  geom_boxplot(alpha=1.0) +
  myTheme + 
  #theme(legend.position = c(0.15,0.7)) +
  theme(legend.position="bottom", legend.title = element_blank()) + 
  ylab('Time Per Epoch (seconds)') + 
  xlab('Number of Distributions') +
  #scale_y_continuous(trans='log2', labels=scaleFUN) +
  scale_fill_discrete(labels = c("EMD", "PairwiseWass"))
  #guides(fill="none", alpha="none")
gg
ggsave("results/wasbary_comparisons_1.pdf", height = 4, width = 6, units = "in")


dorder = c("2", "5", "10", "20", "50", "100")
othistdata$d <- factor(as.character(othistdata$d), levels = dorder)
#torder = c("demd", "sink_1d_bary", "lp_1d_bary", "cvx")
torder = c("demd", "pairwass")
othistdata$model <- factor(as.character(othistdata$model), levels = torder)

gg <- ggplot(othistdata, aes(x=d, y=loss, fill=distType)) + 
  geom_boxplot(alpha=1.0) +
  myTheme + 
  #theme(legend.position = c(0.15,0.7)) +
  theme(legend.position="bottom", legend.title = element_blank()) + 
  ylab('Loss (Distance)') + 
  xlab('Number of Distributions') +
  #scale_y_continuous(trans='log2', labels=scaleFUN) +
  scale_fill_discrete(labels = c("EMD", "PairwiseWass"))
#guides(fill="none", alpha="none")
gg
ggsave("results/wasbary_comparisons_1.pdf", height = 4, width = 6, units = "in")




#############



####speed histogram
lossdata = data.frame(read.csv("results/2_tmpres.csv"))

dorder = c("2", "5", "10", "20", "50", "100")
othistdata$d <- factor(as.character(othistdata$d), levels = dorder)
#torder = c("demd", "sink_1d_bary", "lp_1d_bary", "cvx")
torder = c("demd", "pairwass")
othistdata$model <- factor(as.character(othistdata$model), levels = torder)

gg <- ggplot(othistdata, aes(x=d, y=time_per_epoch, fill=distType)) + 
  geom_boxplot(alpha=1.0) +
  myTheme + 
  #theme(legend.position = c(0.15,0.7)) +
  theme(legend.position="bottom", legend.title = element_blank()) + 
  ylab('Time Per Epoch (seconds)') + 
  xlab('Number of Distributions') +
  #scale_y_continuous(trans='log2', labels=scaleFUN) +
  scale_fill_discrete(labels = c("EMD", "PairwiseWass"))
#guides(fill="none", alpha="none")
gg
ggsave("results/wasbary.pdf", height = 4, width = 6, units = "in")





