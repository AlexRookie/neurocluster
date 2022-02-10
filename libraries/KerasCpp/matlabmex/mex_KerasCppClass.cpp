/* ======================================================================== *\
Mex file for RubiksCubeClass
Alessandro Antonucci and Davide Vignotto
University of Trento
\* ======================================================================== */

// Libraries
#include <vector>
//#include <string> 
//#include <map>

#include "mex_utils.hpp"
#include "../KerasCppClass.hpp"

// Namespace
using namespace std;

/*==========================================================================*\
Mex Function
\*==========================================================================*/

static
void
DATA_NEW( mxArray * & mx_id ) {
  KerasCpp *ptr = new KerasCpp();
  mx_id = convertPtr2Mat<KerasCpp>(ptr);
}

static
inline
KerasCpp *
DATA_GET( mxArray const * & mx_id ) {
  return convertMat2Ptr<KerasCpp>(mx_id);
}

// Callback for methods

static
void
do_new( int nlhs, mxArray       *plhs[],
        int nrhs, mxArray const *prhs[] ) {

  DATA_NEW(arg_out_0); // get OBJ
}

static
void
do_delete( int nlhs, mxArray       *plhs[],
           int nrhs, mxArray const *prhs[] ) {

  KerasCpp *ptr = DATA_GET(arg_in_1); // get OBJ
  if ( ptr != nullptr ) delete ptr;
}

static
void
do_predict( int nlhs, mxArray       *plhs[],
            int nrhs, mxArray const *prhs[] ) {

  KerasCpp *ptr = DATA_GET(arg_in_1); // get OBJ

  // arguments
  mwSize sz;
  const double *matlab_in_vector = getVectorPointer(arg_in_2, sz, "Input expected be a double vector");
  std::vector<double> input(sz);
  for (mwSize i=0; i < sz; ++i) {
    input[i] = matlab_in_vector[i];
  }

  std::vector<double> output;
  ptr -> predict(input, output);

  // output
  double *matlab_out_vector = createMatrixValue(arg_out_0, 1, output.size());
  for (int i=0; i < output.size(); i++) {
    matlab_out_vector[i] = output[i];
  }
}

// List of methods to be mapped

typedef enum {
  CMD_NEW,
  CMD_DELETE,
  CMD_PREDICT
} CMD_LIST;

// Use MAP to connect mapped methods with strings

static map<string,unsigned> cmd_to_idx = {
  {"new",CMD_NEW},
  {"delete",CMD_DELETE},
  {"predict",CMD_PREDICT}
};

// MEX

extern "C"
void
mexFunction( int nlhs, mxArray       *plhs[],
             int nrhs, mxArray const *prhs[] ) {
  // the first argument must be a string
  if ( nrhs == 0 ) {
    mexErrMsgTxt("Erro: first argument must be a string!!");
    return;
  }

  try {

    MEX_ASSERT( mxIsChar(arg_in_0), "First argument must be a string" );
    string cmd = mxArrayToString(arg_in_0);

    switch ( cmd_to_idx.at(cmd) ) {
    case CMD_NEW:
      do_new( nlhs, plhs, nrhs, prhs );
      break;
    case CMD_DELETE:
      do_delete( nlhs, plhs, nrhs, prhs );
      break;
    case CMD_PREDICT:
      do_predict( nlhs, plhs, nrhs, prhs );
      break;
    }
  } catch ( exception const & e ) {
    string err = mxArrayToString(arg_in_0);
    err += "\n";
    err += e.what();
    mexErrMsgTxt(err.c_str());
  } catch (...) {
    mexErrMsgTxt("KerasCppClass failed, unknown error\n");
  }

}
