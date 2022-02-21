#pragma once

#include <vector>
#include "circular_buffer.hh"

const int WINDOW = 12;
const int CLASSES = 3;
const int ROWS = 5;
const int COLS = WINDOW;

bool got_pred = false;
jm::circular_buffer<double, WINDOW> x, y, tcos, tsin, kappa;
double in_nn[ROWS][COLS];
double out_nn[CLASSES];