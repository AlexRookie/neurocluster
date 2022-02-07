#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "KerasCppClass.hpp"
#include "appo_model_weights.hpp"
#include "som_network.hpp"

#include <iostream>

// Constructor
KerasCpp::KerasCpp () {
}

// Destructor
KerasCpp::~KerasCpp() {
}

std::vector<double> KerasCpp::predict(std::vector<double> nn_in) {
  double arr_in[UNITS];
  double arr_out[CLASSES];
  std::vector<double> nn_out(CLASSES);
  
  for (int i=0; i<UNITS; i++) {
    arr_in[i] = nn_in[i];
    std::cout << arr_in[i] << " " << nn_in[i] << std::endl;
  } 

  Som<SOM_ROWS, SOM_COLS, UNITS, CLASSES>(arr_in, arr_out);

  for (int i=0; i<CLASSES; i++) {
    nn_out[i] = arr_out[i];
    std::cout << nn_out[i] << " " << arr_out[i] << std::endl;
  }
  return nn_out;
}