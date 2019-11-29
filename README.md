#Assignment 3:

The value of Validation accuracy after 50 epochs of the already given network - val_acc: 0.8303

The value of final Validation accuracy after50 epochs when the network is modified - val_acc: 0.8259 (44th Epoch)

Model Definition:

# Define the model
model_input = Input((32,32,3))
model2 = SeparableConv2D(40, (3,3), padding='same') (model_input)  #Output:32,32,40   RF:3
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model2 = SeparableConv2D(40, (3,3)) (model2)  #Output:30,30,40   RF:5
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model = SeparableConv2D(40, (3,3), padding='same') (model2)  #Output:30,30,40   RF:7
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model2 = MaxPooling2D() (model2)  #Output:15,15,40   RF:8
model2 = Dropout(0.15) (model2)

model2 = SeparableConv2D(80, (3,3), padding='same') (model2)  #Output:15,15,80   RF:12
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model2 = SeparableConv2D(80, (3,3)) (model2)  #Output:13,13,80   RF:16
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model2 = SeparableConv2D(80, (3,3), padding='same') (model2)  #Output:13,13,80   RF20
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model2 = MaxPooling2D() (model2)  #Output:6,6,80   RF:22
model2 = Dropout(0.15) (model2)

model2 = SeparableConv2D(160, (3,3), padding='same') (model2)  #Output:6,6,160   RF:30
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model2 = SeparableConv2D(160, (3,3)) (model2)  #Output:4,4,160   RF:38
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model2 = SeparableConv2D(160, (3,3), padding='same') (model2) #Output:4,4,160   RF:46
model2 = BatchNormalization() (model2)
model2 = Activation('relu') (model2)

model2 = MaxPooling2D() (model2)  #Output:2,2,160   RF:50
model2 = Dropout(0.15) (model2)

model2 = SeparableConv2D(10, (2,2)) (model2)  #Output:1,1,10   RF:58
model2 = Flatten() (model2)
model2 = Activation('softmax') (model2)

model2 = Model(inputs = model_input, outputs = model2)

# Compile the model
def scheduler(epoch, lr):
  return round(0.003 * 1/(1+0.319 * epoch),10)
model2.compile(optimizer=Adam(0.003), loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()


#Logs:

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
1562/1562 [==============================] - 39s 25ms/step - loss: 1.4296 - acc: 0.4853 - val_loss: 1.8378 - val_acc: 0.4409
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
1562/1562 [==============================] - 37s 23ms/step - loss: 1.0507 - acc: 0.6279 - val_loss: 0.9639 - val_acc: 0.6652
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.9049 - acc: 0.6820 - val_loss: 1.0389 - val_acc: 0.6516
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.8182 - acc: 0.7139 - val_loss: 1.0685 - val_acc: 0.6456
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.7668 - acc: 0.7340 - val_loss: 0.8284 - val_acc: 0.7168
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.7199 - acc: 0.7472 - val_loss: 0.9719 - val_acc: 0.6771
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.6895 - acc: 0.7607 - val_loss: 0.8482 - val_acc: 0.7211
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
1562/1562 [==============================] - 37s 24ms/step - loss: 0.6635 - acc: 0.7690 - val_loss: 0.7707 - val_acc: 0.7415
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.6443 - acc: 0.7755 - val_loss: 0.7547 - val_acc: 0.7439
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
1562/1562 [==============================] - 37s 24ms/step - loss: 0.6203 - acc: 0.7854 - val_loss: 0.6557 - val_acc: 0.7824
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.6046 - acc: 0.7897 - val_loss: 0.6793 - val_acc: 0.7765
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.5880 - acc: 0.7958 - val_loss: 0.7051 - val_acc: 0.7671
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.5792 - acc: 0.7987 - val_loss: 0.6505 - val_acc: 0.7790
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.5644 - acc: 0.8039 - val_loss: 0.6888 - val_acc: 0.7747
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.5546 - acc: 0.8067 - val_loss: 0.5900 - val_acc: 0.8039
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
1562/1562 [==============================] - 37s 24ms/step - loss: 0.5450 - acc: 0.8109 - val_loss: 0.6431 - val_acc: 0.7883
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.5340 - acc: 0.8134 - val_loss: 0.6515 - val_acc: 0.7819
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.5252 - acc: 0.8152 - val_loss: 0.6674 - val_acc: 0.7792
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.5203 - acc: 0.8181 - val_loss: 0.6465 - val_acc: 0.7826
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.5098 - acc: 0.8231 - val_loss: 0.6018 - val_acc: 0.7998
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.5081 - acc: 0.8205 - val_loss: 0.6189 - val_acc: 0.8013
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4962 - acc: 0.8254 - val_loss: 0.6159 - val_acc: 0.8006
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4940 - acc: 0.8257 - val_loss: 0.6329 - val_acc: 0.7933
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4901 - acc: 0.8281 - val_loss: 0.5855 - val_acc: 0.8101
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4825 - acc: 0.8332 - val_loss: 0.6133 - val_acc: 0.7993
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4838 - acc: 0.8291 - val_loss: 0.6174 - val_acc: 0.8037
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4707 - acc: 0.8358 - val_loss: 0.5831 - val_acc: 0.8114
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
1562/1562 [==============================] - 38s 24ms/step - loss: 0.4726 - acc: 0.8341 - val_loss: 0.5554 - val_acc: 0.8160
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4654 - acc: 0.8359 - val_loss: 0.5805 - val_acc: 0.8103
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4610 - acc: 0.8389 - val_loss: 0.6231 - val_acc: 0.8019
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4603 - acc: 0.8394 - val_loss: 0.6192 - val_acc: 0.8026
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4611 - acc: 0.8373 - val_loss: 0.5897 - val_acc: 0.8104
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4533 - acc: 0.8420 - val_loss: 0.5533 - val_acc: 0.8168
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4446 - acc: 0.8438 - val_loss: 0.5573 - val_acc: 0.8207
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4523 - acc: 0.8406 - val_loss: 0.5579 - val_acc: 0.8180
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4455 - acc: 0.8443 - val_loss: 0.5802 - val_acc: 0.8146
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4399 - acc: 0.8461 - val_loss: 0.5658 - val_acc: 0.8167
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4451 - acc: 0.8437 - val_loss: 0.5714 - val_acc: 0.8142
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4348 - acc: 0.8474 - val_loss: 0.5622 - val_acc: 0.8182
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4322 - acc: 0.8476 - val_loss: 0.5736 - val_acc: 0.8160
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4305 - acc: 0.8472 - val_loss: 0.5565 - val_acc: 0.8197
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4308 - acc: 0.8483 - val_loss: 0.5821 - val_acc: 0.8150
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4254 - acc: 0.8496 - val_loss: 0.5858 - val_acc: 0.8150
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4199 - acc: 0.8517 - val_loss: 0.5407 - val_acc: 0.8259
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4226 - acc: 0.8511 - val_loss: 0.5549 - val_acc: 0.8217
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4240 - acc: 0.8521 - val_loss: 0.5674 - val_acc: 0.8189
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4199 - acc: 0.8535 - val_loss: 0.5515 - val_acc: 0.8212
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
1562/1562 [==============================] - 36s 23ms/step - loss: 0.4164 - acc: 0.8539 - val_loss: 0.5714 - val_acc: 0.8160
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
1562/1562 [==============================] - 37s 23ms/step - loss: 0.4176 - acc: 0.8531 - val_loss: 0.5575 - val_acc: 0.8221
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
1562/1562 [==============================] - 37s 24ms/step - loss: 0.4146 - acc: 0.8541 - val_loss: 0.5704 - val_acc: 0.8199
Model took 1829.49 seconds to train

