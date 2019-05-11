
See http://www.vlfeat.org/matconvnet/mfiles/simplenn/vl_simplenn
for an explanation of the parameters of the layers in the network


FULLY CONNECTED LAYERS HAVE NO SPECIFIC NAME: THEY CAN BE IMPLEMENTED
WITH CONVOLUTIONAL LAYERS ('conv') AND WIDTH/HEIGHT EQUAL TO THE
OUTPUT OF THE PREVIOUS LAYER
(see https://github.com/vlfeat/matconvnet/issues/185)

---

Interpretation of the printed output:

  When CNNopts.errorFunction = 'multiclass'

    objective is the value of the loss function
    top1err is the error in the first prediciton: 1-accuracy
    top5err is the error in the five first predictions

  When CNNopts.errorFunction = 'binary'

    objective is the value of the loss function
    binerror is the error in the prediciton: 1-accuracy

