#pragma once

#include <vector>
#include "circular_buffer.hh"

const int WINDOW = 20;
const int UNITS = WINDOW*4;
const int CLASSES = 3;
const int SOM_ROWS = 10;
const int SOM_COLS = 10;

bool got_pred = false;
jm::circular_buffer<double, WINDOW> x, y, theta, kappa;
double in_nn[UNITS];
double out_nn[CLASSES];