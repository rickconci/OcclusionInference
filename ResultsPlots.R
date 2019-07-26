library(ggplot2)


## SUPERVIED ENCODER 
My_title ='Supervised encoder accuracies'
models = c("B", "B_matched","BL", "BT", "BLT","BLT_matched","BLT_higher")
gnrl_depth_accuracy = c( )
gnrl_black_accuracy = c(86.2, 89.9, 93.9, 94.1, 95.4, 90.5, 93.5 )
gnrl_white_accuracy = c( 85.8, 88.5, 94.7, 94.0, 93.5, 90.3, 95.5)
df <- data.frame(Model=c(models, models), Y = c(gnrl_black_accuracy,gnrl_white_accuracy),
                 type = c(rep("black digit", length(models)), rep("white digit", length(models))) )



## SUPERVIED DECODER
My_title ='Supervised decoder reconstruction loss'
models = c("B", "B_matched","BL", "BT", "BLT")
test_recon_loss = c(12.9, 11.5, 8.9, 8.2, 7.8)
gnrl_recon_loss = c(16.4, 16.1, 19.4, 18.7, 18.8 )
#test_recon_loss = c(  , , , 8.5,  8.3)
#gnrl_recon_loss = c(  , , , 9.8, 9.5 )
df <- data.frame(Model=c(models, models), Y = c(test_recon_loss,gnrl_recon_loss),
                 type = c(rep("test recon loss", length(models)), rep("gnrl recon loss", length(models))) )



## UNSUPERVISED 
My_title ='Unsupervised reconstruction loss'
models = c("B-B", "B_matched-B","BL-B", "BT-B", "BLT-B")
test_recon_loss = c(29.5, 24.1, 31.4, 23.5,  23.3)
gnrl_recon_loss = c(34.0, 26.3, 35, 25.2, 25.0 )
df <- data.frame(Model=c(models, models), Y = c(test_recon_loss,gnrl_recon_loss),
                  type = c(rep("test recon loss", length(models)), rep("gnrl recon loss", length(models))) )

  
ggplot(data=df, aes(x=factor(Model,levels=unique(Model)), y=Y, fill=factor(type,levels=unique(type)))) +
  geom_bar(stat="identity", position=position_dodge()) + 
  scale_fill_brewer(palette="Paired") +  theme_minimal() +
  geom_text(aes(label=round(Y,digits=2)), vjust=1.6, color="black",
            position = position_dodge(0.9), size=3.5)+
  theme(axis.text.x = element_text(size = rel(1.1), angle = 45, margin = margin(t = 20)),
        axis.text.y = element_text(size = rel(1.1)),
        axis.title =  element_text(size = rel(1.1)),
        plot.title = element_text(hjust = 0.5,face = "bold") ) +
  labs(title = My_title[1], fill = '') + ylab('Loss') + xlab('Models') #+ 
  #coord_cartesian(ylim = c(80, 100)) 
 

