## Karar Ağaçları (Decision Trees)##
install.packages("rpart")
library(rpart) #Tahmin edilen değişken faktör ya da nümerik verilmeli. Nümerikse karar ağaçlarını faktörse classification yapar.
.
install.packages("rattle")
library(rattle) #görselleştirme için

breast_cancer <- read.csv('breast_cancer.csv' , header = TRUE , sep = "," , dec = ".")
View(breast_cancer)

#ID Number: Her bir gözlem için benzersiz bir tanımlayıcı.
#Diagnosis: Teşhis sonucu. İki kategori içerir:
#B: Benign (iyi huylu)
#M: Malignant (kötü huylu)
#Radius: Çekirdek yarıçapı (çapın yarısı).
#Texture: Çekirdek yüzeyinin pürüzlülüğü.
#Perimeter: Çekirdek çevresi.
#Area: Çekirdek alanı.
#Smoothness: Çekirdek yüzeyindeki pürüzsüzlük derecesi (kenarın uzunluğuna göre lokal değişiklikler).
#Compactness: Çekirdek şekli kompaktlığı (çevrenin alan karesine oranı).
#Concavity: Çekirdek sınırlarının girintili olma derecesi.
#Concave points: Çekirdek konturundaki girintili noktaların sayısı.
#Symmetry: Çekirdek simetrisi.
#Fractal dimension: Çekirdek sınırının pürüzlülüğü (çevre / alan).

length(breast_cancer)
cancer<-breast_cancer[,-c(1,33)]

View(cancer)
nrow(cancer)

library(mice)
md.pattern(cancer)


table(cancer$diagnosis)

set.seed(145)
train <- sample(1:nrow(cancer) , size = 0.8*nrow(cancer))

trainSet <- cancer[train,]
testSet <- cancer[-train,]

nrow(trainSet)
nrow(testSet)
table(trainSet$diagnosis)
table(testSet$diagnosis)

### Model Oluşturma

trainSet$diagnosis <- as.factor(trainSet$diagnosis)
testSet$diagnosis <- as.factor(testSet$diagnosis)


#information entropy demektir
model_Entropy <- rpart(diagnosis ~ . , data = trainSet , method = "class" , 
                      parms = list(split = "information"))

model_Gini <- rpart(diagnosis~ . , data = trainSet , method = "class" , 
                   parms = list(split = "gini"))

model_Entropy_1 <- rpart(diagnosis ~ . , data = trainSet , method = "class" , 
                       parms = list(split = "information"))

model_Entropy # Yıldızlı olanlar terminal/leaf node. En son karar verdiğimiz yer

model_Gini

model_Entropy_1

### Karar Agacının Görselleştirilmesi

fancyRpartPlot(model_Entropy)
fancyRpartPlot(model_Gini)
fancyRpartPlot(model_Entropy_1)



### Model Detayları

summary(model_Entropy)
summary(model_Gini)
summary(model_Entropy_1)

### Karar Ağaçları Hiper Parametreleri


model_Hyper_Parameter<- rpart(diagnosis ~ . , data = trainSet , method = "class" , 
                           parms = list(split = "information") ,
                           control = rpart.control(minsplit = 40 , cp = 0.02 , maxdepth = 5))

model_Hyper_Parameter

fancyRpartPlot(modelEntropyHyper)


### Tahmin

#test veri setindeki tahminlerle karşılaştırma yaparız
#Çıktıları olasılık olarak değil class olarak almak istersek type=class olur

pred_Entropy <- predict(model_Entropy , testSet , type = "class")
pred_Gini <- predict(model_Gini , testSet , type = "class")
pred_Entropy_Hyper <- predict(model_Hyper_Parameter , testSet , type="class")

library(caret)

#hagisi daha iyi?

confusionMatrix(pred_Entropy , testSet$diagnosis) # B=iyi huylu tümör için tahmin başarısı
confusionMatrix(pred_Entropy , testSet$diagnosis , mode = "prec_recall") # B=iyi huylu tümör için tahmin başarısı için F1 skorları
confusionMatrix(pred_Entropy , testSet$diagnosis , mode = "prec_recall" , positive = "M") # M=Kötü huylu tümör için tahmin başarısı


confusionMatrix(pred_Gini , testSet$diagnosis)
confusionMatrix(pred_Gini , testSet$diagnosis , mode = "prec_recall")
confusionMatrix(pred_Gini , testSet$diagnosis , mode = "prec_recall" , positive = "M")


confusionMatrix(pred_Entropy_Hyper , testSet$diagnosis)
confusionMatrix(pred_Entropy_Hyper , testSet$diagnosis , mode = "prec_recall") 
confusionMatrix(pred_Entropy_Hyper , testSet$diagnosis , mode = "prec_recall" , positive = "M")

## Model Tuning

#Hiperparametre tuning işlemi, bu ayarların farklı değerlerini deneyerek en iyi sonuçları veren kombinasyonu bulmaya çalışılır. 
#Bunun amacı, modelin doğruluğunu arttırmak, aşırı öğrenme(overfitting) veya yetersiz  öğrenme (underfitting) gibi sorunlardan kaçınmaktır.


modelLookup("rpart") # cp'yi tune ettiğini gösterir
modelLookup("rpart2") #rpart2'nin maxdepth'i tune ettiğini gC6sterir.
library(e1071) #tune.rpart() ile diğer parametreleri de tune edebiliriz.

Control <- trainControl(method="cv" , number = 5 , search = "random") # number k sayısıdır ve kaç parçaya böleceği, search=random olarak cp'yi bulsun.
Control1 <- trainControl(method="cv" , number = 5 , search = "grid") # grid yöntemiyle yap

#rpart metodu ile
modelCP <- train(diagnosis ~ . , data  = trainSet ,
                 method = "rpart" ,
                 tuneLength = 20,
                 trControl = Control
)
modelCP
# en yüksek accuracy için cp değerini verir.


#rpart2 metodu ile
modelMD <- train(diagnosis ~ . , data  = trainSet ,
                 method = "rpart2" ,
                 tuneLength = 20,
                 trControl = Control
)
modelMD
#en yüksek accuracy için en yüksek doğruluğu veren depth maxdepth olur.



#maxdepth biz veriyoruz ve hepsini kontrol eder
modelMDGrid <- train(diagnosis~ . , data  = trainSet ,
                     method = "rpart2" ,
                     tuneGrid = expand.grid(maxdepth = 3:20),
                     trControl = Control1
)
modelMDGrid

# e1071() iC'indeki tune.rpart() ile,
modelTuneMin <- tune.rpart(diagnosis ~ . , data  = trainSet ,
                           minsplit = 10:15 , minbucket = 5:10 , cp = seq(0.0 , 0.2 , by = 0.01))
modelTuneMin

modeltuneMax<-tune.rpart(diagnosis~ . , data  = trainSet ,maxdepth = 5,
                         minsplit = 10:15 , minbucket = 5:10 , cp = seq(0.0 , 0.2 , by = 0.01))
modeltuneMax
# sonuçta çıkan değerler modelde parametre için verdiğimiz değerlere göre çıkar

### Tune Edilmiş Model Üzerinden Tahminler 

modelMDGrid$finalModel


predMDGrid <- predict(modelMDGrid$finalModel , testSet , type = "class")
predCP <- predict(modelCP$finalModel , testSet , type = "class")
predMD <- predict(modelMD$finalModel , testSet , type = "class")
predMin <- predict(modelTuneMin$best.model , testSet , type = "class")
predMax <- predict(modeltuneMax$best.model , testSet , type = "class")


confusionMatrix(predMDGrid  , testSet$diagnosis , mode = "prec_recall" , positive = "M")
confusionMatrix(predCP  , testSet$diagnosis , mode = "prec_recall" , positive = "M")
confusionMatrix(predMD  , testSet$diagnosis , mode = "prec_recall" , positive = "M")
confusionMatrix(predMin  , testSet$diagnosis , mode = "prec_recall" , positive = "M")
confusionMatrix(predMax  , testSet$diagnosis , mode = "prec_recall" , positive = "M")


fancyRpartPlot(modelTuneMin$best.model)
fancyRpartPlot(modeltuneMax$best.model)


### Random Forest Modeli Oluşturma

#install.packages("randomForest")
library(randomForest)

modelRF <- randomForest(diagnosis~ . , data  = trainSet , ntree = 500 ) #ntree modelde kaç ağaç oluşturacağız. default 500dür
#mtry daha sonra tunin olarak kullanacağız bu yüzden defaultta bıraktık. Kaç değişken kullanacağımızı veririz

modelRF$err.rate # ağaçlardaki hata oranları
#OOB=out of back. B ve M classlara göre ağaçların hatasını veririr

modelRF$mtry # diğer parametreleri kontrol edebiliriz.Ayırma için kaç değişken kullanılacağını gösterir


## Random Forest Tahmin
predRF <- predict(modelRF , testSet)
predRF

library(caret)

confusionMatrix(predRF , testSet$diagnosis)
confusionMatrix(predRF , testSet$diagnosis , mode = "prec_recall" , positive = "M")

## Random Forest Tune İşlemi


modelLookup("rf")

ControlRF <- trainControl(method = "repeatedcv" , 
                            number = 10 , repeats = 3 , search = "random") 


modelRF <- train(diagnosis ~ . , data = trainSet , 
                     method = "rf",
                     tuneLength = 20,
                     trControl  = ControlRF
) 
modelRF


#tune grid de kullanılabilir


ControlRF2 <- trainControl(method = "cv" , number = 5, search = "grid") # Çok beklememek için repeated vermedik



modelRFGrid <- train(diagnosis ~ . , data = trainSet , 
                         method = "rf",
                         tuneGrid = expand.grid(mtry = 1:8),
                         trControl  = ControlRF2
) 

modelRFGrid 

## Tahminler 

predRFTune <- predict(modelRF$finalModel , testSet)

confusionMatrix(predRF , testSet$diagnosis , mode = "prec_recall" , positive = "M") #tune edilmemiş tahmin
confusionMatrix(predRFTune , testSet$diagnosis , mode = "prec_recall" , positive="M")


