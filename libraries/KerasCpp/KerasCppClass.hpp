const int UNITS = 80;
const int CLASSES = 3;
const int SOM_ROWS = 10;
const int SOM_COLS = 10;

#include <vector>

class KerasCpp {

public:

  // Constructor: initialize the state
  KerasCpp();

  // Destructor: clear the instance
  ~KerasCpp();

  void predict(std::vector<double> nn_in, std::vector<double> & nn_out);
};