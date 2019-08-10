library(ggplot2)


## SUPERVIED ENCODER 
My_title ='Supervised encoder accuracies on generalisation set \n dataset solid2: colour binding'
models = c("B", "B_matched","BL", "BT", "BLT")
gnrl_depth_accuracy = c( )
gnrl_black_accuracy = c(86.5, 89.9, 93.9, 94.1, 95.4 )
gnrl_white_accuracy = c( 86.3, 88.5, 94.7, 94.0, 93.5)
df <- data.frame(Model=c(models, models), Y = c(gnrl_black_accuracy,gnrl_white_accuracy),
                 type = c(rep("black digit", length(models)), rep("white digit", length(models))) )


My_title ='Supervised encoder accuracies on generalisation set \n dataset solid2: depth binding'
models = c("B", "B_matched","BL", "BT", "BLT")
gnrl_back_accuracy = c(55.1, 46.2, 66.8, 59.8, 56.3 )
gnrl_front_accuracy = c( 99.6, 99.7, 99.6, 99.3, 98.4)
df <- data.frame(Model=c(models, models), Y = c(gnrl_back_accuracy,gnrl_front_accuracy),
                 type = c(rep("back digit", length(models)), rep("front digit", length(models))) )



My_title ='Supervised encoder accuracies dataset \n border2: depth binding'
models = c("B", "B_matched","BL", "BT", "BLT")
gnrl_back_accuracy = c(34.1, 35.0, 47.2, 47.9, 50.3 )
gnrl_front_accuracy = c( 98.6, 98.1, 98.4, 98.2, 99.0)
df <- data.frame(Model=c(models, models), Y = c(gnrl_back_accuracy,gnrl_front_accuracy),
                 type = c(rep("back digit", length(models)), rep("front digit", length(models))) )




My_title ='Supervised encoder accuracies on generalisation set \n border3 dataset: depth binding'
models = c("B", "B_matched","BL", "BT", "BLT")
gnrl_back_accuracy = c(26.0, 35.5,39.3, 40.1, 41.9 )
gnrl_mid_accuracy = c(32.5, 37.1,  38.9, 36.2, 38.7  )
gnrl_front_accuracy = c(93.4,93.9, 94.3,93.4, 94.1)
average_accuracy = (gnrl_back_accuracy + gnrl_mid_accuracy + gnrl_front_accuracy)/3
df <-  data.frame(Model=c(models, models, models), Y = c(gnrl_back_accuracy,gnrl_mid_accuracy, gnrl_front_accuracy),
                  type = c(rep("back digit", length(models)), rep("mid digit", length(models)),  rep("front digit", length(models)) ) )
#df <-  data.frame(Model=models, Y =average_accuracy, type = "all digits", length(models)) 



models = c("B", "B_matched","BL", "BT", "BLT","BLT_matched","BLT_higher")
gnrl_back_accuracy = c(46.5,  59.3, )
gnrl_front_accuracy = c(96.5, 98.4, )
df <-  data.frame(Model=c(models, models), Y = c(gnrl_back_accuracy, gnrl_front_accuracy),
                  type = c(rep("back digit", length(models)),  rep("front digit", length(models)) ) )



## SUPERVIED DECODER
My_title ='Supervised decoder reconstruction loss \n Dataset solid2'
models = c("B", "B_matched","BL", "BT", "BLT")
train_recon_loss = c(11.4, 10.4, 6.2, 6.5, 5.6)
gnrl_recon_loss = c(11.9 , 13.8, 10.2 , 13.7, 15.1 )
df <- data.frame(Model=c(models, models), Y = c(train_recon_loss, gnrl_recon_loss),
                 type = c( rep("train  reconstruction loss", length(models)), rep("gnrl reconstruction loss", length(models))  ) )



My_title ='Supervised decoder reconstruction loss \n Dataset B'
models = c("B", "B_matched","BL", "BT", "BLT")

train_recon_loss = c(32.2, 29.8, 17.1, 22.8, 15.0)
gnrl_recon_loss = c(32.4 , 30.2, 18.3 , 23.7, 16.6 )
df <- data.frame(Model=c(models, models), Y = c(train_recon_loss, gnrl_recon_loss),
                 type = c( rep("train  reconstruction loss", length(models)), rep("gnrl reconstruction loss", length(models))  ) )


My_title ='Supervised decoder reconstruction loss \n Dataset border3'
models = c("B", "B_matched","BL", "BT", "BLT")

train_recon_loss = c(45.7, 43.5, 30.2, 33.4, 26.7)
gnrl_recon_loss = c(45.8 , 46.1, 30.8 , 33.1, 27.1 )
df <- data.frame(Model=c(models, models), Y = c(train_recon_loss, gnrl_recon_loss),
                 type = c( rep("train  reconstruction loss", length(models)), rep("gnrl reconstruction loss", length(models))  ) )




## UNSUPERVISED 
My_title ='Unsupervised reconstruction loss'
models = c("B-B", "B_matched-B","BL-B", "BT-B", "BLT-B")
test_recon_loss = c(29.5, 24.1, 31.4, 23.5,  23.3)
gnrl_recon_loss = c(34.0, 26.3, 35, 25.2, 25.0 )
df <- data.frame(Model=c(models, models), Y = c(test_recon_loss,gnrl_recon_loss),
                  type = c(rep("test recon loss", length(models)), rep("gnrl recon loss", length(models))) )
  

My_title ='Range of activaitions \n changing code layer size: border2'
z_dim = c("6", "12","18", "24", "30")
range.act = c(28.0, 19.2, 16.6, 15.5, 14.7 )
df <- data.frame(Model=c(z_dim), Y = c(range.act),
                 type = rep("train reconstruction loss", length(z_dim)) )



My_title ='Range of activaitions \n changing code layer size: border3'
z_dim = c("6", "12","18", "24", "30")
range.act= c(20.7, 13.5, 11.5, 11.3, 10.4)
df <- data.frame(Model=c(z_dim), Y = c(range.act),
                 type = rep("train reconstruction loss", length(z_dim)) )



My_title ='Unsupervised reconstruction loss \n changing code layer size: border2'
z_dim = c("6", "12","18", "24", "30")
train_recon_loss= c(20.7, 13.5, 11.5, 11.3, 10.4)
gnrl_recon_loss = c(28.0, 19.2, 16.6, 15.5, 14.7 )
df <- data.frame(Model=c(z_dim, z_dim), Y = c(train_recon_loss,gnrl_recon_loss),
                 type = c(rep("train reconstruction loss", length(z_dim)), rep("gnrl reconstruction loss", length(z_dim))) )


My_title ='Unsupervised reconstruction loss \n changing code layer size: border3'
z_dim = c("6", "12","18", "24", "30", "36")
train_recon_loss= c(47.3, 31.9, 28.7, 28.2, 24.8, 25.0 )
gnrl_recon_loss = c(53.8, 41.8, 37.7, 35.6, 32.9, 32.4 )
df <- data.frame(Model=c(z_dim, z_dim), Y = c(train_recon_loss,gnrl_recon_loss),
                 type = c(rep("train reconstruction loss", length(z_dim)), rep("gnrl reconstruction loss", length(z_dim))) )



ggplot(data=df, aes(x=factor(Model,levels=unique(Model)), y=Y, fill=factor(type,levels=unique(type)))) +
  geom_bar(stat="identity", position=position_dodge()) + 
  scale_fill_brewer(palette="Blues") +  theme_minimal() +
  geom_text(aes(label=round(Y,digits=2)), vjust=1.6, color="black",
            position = position_dodge(0.9), size=10)+
  theme(axis.text.x = element_text(size = 25, angle = 45, margin = margin(t = 20)),
        axis.text.y = element_text(size = 25),
        axis.title =  element_text(size = 25),
        plot.title = element_text(size = 25, hjust = 0.5,face = "bold"),
        legend.text = element_text(size=25)) +  theme(legend.position="top")+
  labs(title = My_title[1], fill = '') + ylab('Loss') + xlab('Model') #+ 
  #coord_cartesian(ylim = c(80, 100)) 
  


