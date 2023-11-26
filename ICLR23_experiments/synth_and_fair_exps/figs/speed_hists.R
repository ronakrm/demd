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
                legend.position = c(0.15,0.2),
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
speedhistdata = data.frame(read.csv("results/speed_test_results.csv"))

speeddata = speedhistdata
#speeddata = subset(speedhistdata, gradType!='torchdual')
#speeddata = subset(speeddata, gradType!='autograd')
#speeddata = subset(speedhistdata, gradType!='npdual')
#speeddata = subset(speeddata, gradType!='scipy')
speeddata = subset(speeddata, n=10)
#speeddata = subset(speeddata, d<25)

dorder = c("2", "5", "10", "20", "50", "100")
speeddata$d <- factor(as.character(speeddata$d), levels = dorder)
torder = c("scipy", "npdual", "autograd", "torchdual")
speeddata$gradType <- factor(as.character(speeddata$gradType), levels = torder)

scipy = subset(speeddata, gradType!='torchdual')
scipy = subset(scipy, gradType!='autograd')

torch = subset(speeddata, gradType!='npdual')
torch = subset(torch, gradType!='scipy')

plotforw <- function(data, ylab) {
  gg <- ggplot(data, aes(x=d, y=forward_time, fill=gradType)) + 
    geom_boxplot(alpha=1.0) +
    myTheme + 
    theme(legend.position = c(0.15,0.8)) +
    ylab(ylab) + 
    xlab('Number of Distributions') +
    scale_y_continuous(trans='log2', labels=scaleFUN)
    #guides(fill="none", alpha="none")
  return(gg)
}

plotback <- function(data, ylab) {
  gg <- ggplot(data, aes(x=d, y=backward_time, fill=gradType)) + 
    geom_boxplot(alpha=1.0) +
    myTheme + 
    theme(legend.position = c(0.15,0.8)) +
    ylab(ylab) + 
    xlab('Number of Distributions') +
    scale_y_continuous(trans='log2', labels=scaleFUN)
    #guides(fill="none", alpha="none")
  return(gg)
}

gg <- plotforw(torch, 'Forward Time (s)')
gg
ggsave("results/Torch_Forwards_10Bins.pdf", height = 4, width = 6, units = "in")

gg <- plotforw(scipy, 'Forward Time (s)')
gg
ggsave("results/Numpy_Forwards_10Bins.pdf", height = 4, width = 6, units = "in")


gg <- plotback(torch, 'Backward Time (s)')
gg
ggsave("results/Torch_Backward_10Bins.pdf", height = 4, width = 6, units = "in")

gg <- plotback(scipy, 'Backward Time (s)')
gg
ggsave("results/Numpy_Backward_10Bins.pdf", height = 4, width = 6, units = "in")
