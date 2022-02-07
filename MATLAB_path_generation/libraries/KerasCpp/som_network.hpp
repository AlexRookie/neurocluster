// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef SOMLAYER_H
#define SOMLAYER_H

#include "kerascpp/kerascpp.hpp"

#include <iostream>

// SOM-classification network

template <int som_rows, int som_cols, int units, int classes>
void Som(double input[units], double out_classes[classes]) {

    double som_out[som_rows*som_cols];
    som<units,som_rows*som_cols>(input, model_appo_w1, som_out);

    double classifier[classes];
    dense<som_rows*som_cols,classes>(som_out, model_appo_w2, model_appo_b2, classifier);
    softmax<classes>(classifier, out_classes);

    for (int i=0; i<classes; i++) {
        std::cout << out_classes[i] << std::endl;
  }
}

#endif // SOMLAYER_H