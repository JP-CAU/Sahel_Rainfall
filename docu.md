## Introduction
### Aufgabe
Die Aufgabe des Projekts war es, den Regenfall in der Sahel Region in Afrika möglichst präzise vorherzusagen. Die Vorhersage soll dabei für verschiedene Lead Times getroffen werden. Die "Lead Time" (auch Vorlaufzeit oder Vorauslaufzeit genannt) im Bereich des Zeitreihen-Forecastings bezieht sich auf den Zeitraum zwischen dem Zeitpunkt, zu dem eine Vorhersage erstellt wird, und dem Zeitpunkt, zu dem die Vorhersage tatsächlich relevant wird oder genutzt werden soll. Im Projekt lag der Fokus auf den Lead Times Null, Eins, Drei und Sechs. Als Grundlage für die Vorhersage wurde ein Datensatz bereit gestellt. 

### Datensatz
Bei dem bereitgestellten Datensatz handelt es sich um den CICMOD Datensatz. Der Datensatz besteht aus 1000 Jahren simulierten Klima Daten von State of the Art Klimamodellen (FOCI und CESM). Der Datensatz ist Multivariat und besteht aus 29 Klima Indizes, die jeweils in monatlicher Auflösung simuliert wurden. Der Regenfall in der Sahel Region ist einer dieser 29 Indizes. Konkret liegen also Messungen zu 12.000 Zeitpunkten vor, die jeweils aus 29 Klimaindizes bestehen. Aus Rechnerischen Gründen liegen bei dem CESM Modell 1000 Jahre und bei dem FOCI Modell 999 Jahre Daten vor. 

### Datenvorverarbeitung
Die Zeitreihen Daten aus dem CICMOD Datensatz wurden normalisiert, sodass für alle Klimaindizes gilt, dass der Mittelwert Null und die Standardabweichung Eins ist. Ein weiteres Preprocessing war nicht nötig, da es sich um synthetische Daten handelt, die von vielen typischen Probleme wie fehlenden Werten nicht betroffen sind. Die Datengrundlage die für das Training der Modelle genutzt wurde, unterscheidet sich je nach Lead Time. Für die Lead Time Null wurde der Sahel Rainfall Index aus den Trainingsdaten entfernt. Ursache hierfür ist, dass das Modell bei einer Lead Time von Null den tatsächlichen Wert des Regenfalls bereits kennt und diesen somit einfach ausgeben könnte. Bei allen anderen Lead Times wurde der volle Datensatz mit allen 29 Indizes verwendet. Die meisten der Machine Learning Modelle, die im Rahmen des Projektes getestet wurden, arbeiten mit einem Fenster an vergangenen Werten als jeweiligem Input. Die Größe dieses Fensters bestimmt wie viele vergangene Werte für die aktuelle Vorhersage herbeigezogen werden. Eine Fenstergröße von Zwölf bedeutet beispielweise, dass das Modell den Regenfall auf Basis der Werte des letzten Jahres vorhersagt. Im Rahmen der Datenvorverarbeitung wurden die Daten in entsprechende Fenster eingeteilt.

In der letzten Phase des Projekts wurden verschiedene Optimierungsmaßnahmen durchgeführt, um ein optimales Ergebnis zu erzielen. Eine dieser Maßnahmen war die Feature Selection. In diesem Schritt wurde eine ausgewählte Teilmenge der Indizes verwendet. Auch die Fensterlänge wurde in diesem Schritt optimiert.

## Modell Auswahl
Zu Beginn des Projekts wurden Vorschläge gesammelt, welche Modelle geeignet sein könnten um den Regenfall in der Sahel Region vorherzusagen. Ergebnis dieser Diskussion waren folgende Modelle:
- (lineare) Regression
- Multi Layer Perceptron
- Convolutional Neural Network
- Recurrent Neural Network (mit Attention)
- Long Short Term Memory (mit Attention)
- Hybrid CNN + LSTM
- Gated Recurrent Unit Network
- Echo State Networks
- Random Forest
- ADA Boost
- XG Boost
- Light GBM

### Train/Test Split
Die angegebenen Modelle wurden jeweils auf 80 Prozent (~800 Jahre) des Datensatzes trainiert. Die restlichen 20 Prozent wurden zu gleichen Teilen zum Validieren und Testen verwendet. Insgesamt wurden 80 Prozent zum Training, 10 Prozent zum Validieren und 10 Prozent zum Testen verwendet. Dei Daten wurden vor der Aufteilung nicht vermischt, sodass die ersten 800 Jahre zum Training genutzt wurden die nächsten 100 jahre zum Validieren und die letzten 100 Jahre zum Testen.

## Baseline Ergebnisse
Im ersten Schritt wurde jedes der oben beschriebenen Modell trainiert. Um eine Vergleichbarkeit herzustellen, wurde für keines der Modelle eine Optimierung vorgenommen. Die Konfiguration der Hyperparameter entsprach entweder dem Default der entsprechenden Python Bibliotheken oder wurde basierend auf Erfahrungswerten gesetzt. 
Ziel dieser Phase war es die Modelle zu identifiezieren, die am besten abschneiden, um diese in einem zweiten Schritt zu optimieren. 
Die Ergebnisse der Baseline Modelle sind Tabelle 1 zu entnehmen:

## Optimierung 
Nach der ersten Auswah der Modell wurden die vielversprechendsten Modelle weiter optimiert.

Mögliche Bereiche der Optimierung waren:
- Feature Selection (Auswahl einer Teilmenge von Features)
- Anpassung der Input Fensterlänge
- Hyperparameter Tuning
- Feature Engineering (Erstellung neuer Features)

Zu Beginn wurde schnell deutlich, dass sich die obigen Optimierungsmöglichkeiten gegenseitig beeinflussen. Die Relevanz einzelner Features, welches die Grundlage für dei Auswahl bei der Feature Selection darstellt, ist beispielsweise stark abhängig von der Input Fensterlänge (siehe Abbildung). Die Reihenfolge der Optimierungsschritte spielt somit eine Rolle. Das Vorgehen bei der Optimierung war unterschiedlich.

### Vorgehen Jake
Optimized models:
- GRU
- LSTM
- CNN + LSTM
- XG Boost
The first step was the feature engineering. The idea behind the feature engineering was to provide the model with additional input, that might help improve the forecasting performance. Different new features were tried out and the performance increase/decrease for both climate models and all lead times was compared. As this step can take a lot of time, the performance measures wehre only computed on the most promising type of model (LSTM) the results might have been different when applied to the other models.
The first new feature is the month of year (1-12) as a one hot encoding. This means that there are 12 new binary features for each month of the year. A one menas that it its this month, a zero menas it is not this month. The January would look like this 100000000000. This feature improved the average performance of the LSTM model. 
The second new feature is the month of year (1-12) as a cosine encoding. This means that the number of each month is mapped onto the cosine function. The idea behind this encoding is to keep the cyclical propertyof months, where december (12) should be very close to january (1) which would not be the case if simply the numbers were used. This feature improved the average performance of the LSTM model less, when compared to the one hot encoding.
The last new feature is a group of three features. 
- The first is the average of the last three month. This feature is computed by taking the mean of the last three months. 
- The second is the average of the last three years. This feature is computed by taking the mean of the values for the current month of the last 3 years. 
- The last one is the value of the last month. 
All of these features did not improve the performance of the LSTM model. The result of the feature engineering was to only use the month of year as a one hot encoding as additional input. 

The next steps were done as a sequence for each of the models individually. The first step was the feature selection. For the neural network approaches the feature selection was implemented as a backward selection. Backward selection in feature selection for neural networks, also known as backward elimination, is a method used to reduce the number of input features used in a neural network model. It starts with all available features and iteratively removes one feature at a time from the model until a stopping criterion is met. In this case only the removal of each of the 29 features inidividually was tested. When the removal of a features led to better performance, the feature was marked as bad feature. All of the bad features where removed from training. 

The next step was the adaptation of the input windowsize (sequence length). 
### Vorgehen Jannik


## Ergebnisse der Optimierung

## Explainable AI

