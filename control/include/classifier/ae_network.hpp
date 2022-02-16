// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef AENETWORK_H
#define AENETWORK_H

#include "kerascpp/kerascpp.hpp"

// AE-classification network

template <int units_rows, int units_cols, int classes>
void AEclass(double inputs[units_rows][units_cols], double out_classes[classes]) {

    double prm[units_cols][units_rows];
    permute<units_rows,units_cols>(inputs, prm);

    double conv_1[units_cols-3+1][6];
    conv1d<units_cols,units_rows,6,3>(prm, cross3ae_classifier_w2, cross3ae_classifier_b2, conv_1);
    double conv_2[10-3+1][8];
    conv1d<10,6,8,3>(conv_1, cross3ae_classifier_w3, cross3ae_classifier_b3, conv_2);
    double conv_3[8-3+1][10];
    conv1d<8,8,10,3>(conv_2, cross3ae_classifier_w4, cross3ae_classifier_b4, conv_3);

    double flat[60];
    flatten<6,10>(conv_3, flat);
    double encoded[16];
    dense<60,16>(flat, cross3ae_classifier_w6, cross3ae_classifier_b6, encoded);

    double dns[12];
    dense<16,12>(encoded, cross3ae_classifier_w7, cross3ae_classifier_b7, dns);
    relu<12>(dns, dns);

    double classifier[classes];
    dense<12,classes>(dns, cross3ae_classifier_w8, cross3ae_classifier_b8, classifier);
    softmax<classes>(classifier, out_classes);
}

#endif // AENETWORK_H