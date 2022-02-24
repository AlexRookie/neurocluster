#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "KerasCppClass.hpp"
#include "cross3_ae3_classifier_weights.hpp"
#include "ae_network.hpp"

// Constructor
KerasCpp::KerasCpp () {
}

// Destructor
KerasCpp::~KerasCpp() {
}

void KerasCpp::predict(std::vector<double> nn_in, std::vector<double> & nn_out) {
  double arr_in[AE_ROWS][AE_COLS];
  double arr_out[AE_CLASSES];

  int s=0;
  for (int i=0; i<AE_ROWS; i++) {
    for (int j=0; j<AE_COLS; j++) {
      arr_in[i][j] = nn_in[s];
      s++;
    }
  }

  AEclass<AE_ROWS, AE_COLS, AE_CLASSES>(arr_in, arr_out);

  for (int i=0; i<AE_CLASSES; i++) {
    nn_out.push_back(arr_out[i]);
  }

  /*
  double arr_in[SOM_UNITS];
  double arr_out[SOM_CLASSES];
  
  for (int i=0; i<SOM_UNITS; i++) {
    arr_in[i] = nn_in[i];
    //std::cout << arr_in[i] << " " << nn_in[i] << std::endl;
  } 

  Som<SOM_ROWS, SOM_COLS, SOM_UNITS, SOM_CLASSES>(arr_in, arr_out);

  for (int i=0; i<SOM_CLASSES; i++) {
    nn_out.push_back(arr_out[i]);
    //std::cout << nn_out[i] << " " << arr_out[i] << std::endl;
  }
  */
}