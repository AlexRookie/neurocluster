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

    double conv_1[8][6];
    conv1d<units_cols, units_rows, 6, 5>(prm, cross3_ae3_classifier_w2, cross3_ae3_classifier_b2, conv_1);
    double conv_2[6][8];
    conv1d<8, 6, 8, 3>(conv_1, cross3_ae3_classifier_w3, cross3_ae3_classifier_b3, conv_2);
    double conv_3[4][10];
    conv1d<6, 8, 10, 3>(conv_2, cross3_ae3_classifier_w4, cross3_ae3_classifier_b4, conv_3);

    double flat[40];
    flatten<4, 10>(conv_3, flat);

    double dense_1[20];
    dense<40, 20>(flat, cross3_ae3_classifier_w6, cross3_ae3_classifier_b6, dense_1);
    double dense_2[10];
    dense<20, 10>(dense_1, cross3_ae3_classifier_w7, cross3_ae3_classifier_b7, dense_2);
    double encoded[5];
    dense<10, 5>(dense_2, cross3_ae3_classifier_w8, cross3_ae3_classifier_b8, encoded);

    double classifier[classes];
    dense<5, classes>(encoded, cross3_ae3_classifier_w9, cross3_ae3_classifier_b9, classifier);
    softmax<classes>(classifier, out_classes);
}

#endif // AENETWORK_H