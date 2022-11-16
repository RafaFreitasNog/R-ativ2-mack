#instalando e usando caret
library(caret)

# Ler base de dados iris
data(iris)
str(iris)
View(iris)


# Realizar partição entre base de teste e base de treino
set.seed(3033)
intrain <- createDataPartition(y = iris$Species, p= 0.7, list = FALSE)
treino <- iris[intrain,]
teste  <- iris[-intrain,]

dim(treino)
dim(teste)

# Usar rpart
library(rpart.plot)

# Arvore de decisão
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
dtree_fit_info <- train(Species ~., data = treino, method = "rpart",
                        parms = list(split = "information"),
                        trControl=trctrl,
                        tuneLength = 10)
prp(dtree_fit_info$finalModel, box.palette="Reds", tweak=1.2)

#Matriz Confusao Arvore
test_pred_info<-predict(dtree_fit_info, newdata = teste)
confusionMatrix(test_pred_info,teste$Species)

# Instalar class
library(class)

# Normalização
treino_norm <- scale(treino[, 1:4])
teste_norm <- scale(teste[, 1:4])
print(treino_norm)

#K - vizinhos mais próximos K=1
classif_knn <- knn(train = treino_norm,
                   test = teste_norm,
                   cl = treino$Species,
                   k = 1)
classif_knn

#Matriz confusao KNN
cm <- table(teste$Species, classif_knn)
cm

#Acuracia KNN
erro_de_classificacao <- mean(classif_knn != teste$Species)
print(paste('Acuracia =', 1-erro_de_classificacao))

library(e1071)
library(caTools)
library(gmodels)

# Naive Baynes
set.seed(120)
classifier_cl <- naiveBayes(Species ~ ., data = treino)
classifier_cl

y_pred <- predict(classifier_cl, newdata = teste)
cm <- table(teste$Species, y_pred)
print(cm)

CrossTable(teste$Species, y_pred)

# Arvore de decisão dados norm
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
dtree_fit_info <- train(Species ~., data = treino, method = "rpart",
                        parms = list(split = "information"),
                        trControl=trctrl,
                        tuneLength = 10)
prp(dtree_fit_info$finalModel, box.palette="Reds", tweak=1.2)

#Matriz Confusao Arvore Dados Norm
test_pred_info<-predict(dtree_fit_info, newdata = teste_norm)
confusionMatrix(test_pred_info,teste$Species)

