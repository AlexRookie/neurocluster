
/*
 * Include Files
 *
 */
#if defined(MATLAB_MEX_FILE)
#include "tmwtypes.h"
#include "simstruc_types.h"
#else
#include "rtwtypes.h"
#endif



/* %%%-SFUNWIZ_wrapper_includes_Changes_BEGIN --- EDIT HERE TO _END */
// Alessandro Antonucci @AlexRookie
// University of Trento

#include <math.h>
#include <vector>

#include "circular_buffer.hh"
#include "appo_model_weights.hpp"
#include "som_network.hpp"

//#include <iostream>
//#define DEBUG

const int UNITS = 80;
const int units = UNITS/4;
const int CLASSES = 3;
const int SOM_ROWS = 10;
const int SOM_COLS = 10;

bool got_pred;
jm::circular_buffer<double, units> x, y, theta, kappa;
double in_nn[UNITS];
double out_nn[CLASSES];
/* %%%-SFUNWIZ_wrapper_includes_Changes_END --- EDIT HERE TO _BEGIN */
#define u_width 1
#define y_width 1

/*
 * Create external references here.  
 *
 */
/* %%%-SFUNWIZ_wrapper_externs_Changes_BEGIN --- EDIT HERE TO _END */
/* extern double func(double a); */
/* %%%-SFUNWIZ_wrapper_externs_Changes_END --- EDIT HERE TO _BEGIN */

/*
 * Start function
 *
 */
void classifier_Start_wrapper(void)
{
/* %%%-SFUNWIZ_wrapper_Start_Changes_BEGIN --- EDIT HERE TO _END */
got_pred = false;
x.clear();
y.clear();
theta.clear();
kappa.clear();
/* %%%-SFUNWIZ_wrapper_Start_Changes_END --- EDIT HERE TO _BEGIN */
}
/*
 * Output function
 *
 */
void classifier_Outputs_wrapper(const real_T *time,
			const real_T *in_x,
			const real_T *in_y,
			const real_T *in_theta,
			const real_T *in_kappa,
			real_T *out_conf,
			int32_T *out_class,
			boolean_T *out_flag)
{
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_BEGIN --- EDIT HERE TO _END */
#ifdef DEBUG
std::cout << "--------------------------------------------------\n";
std::cout << "clock: " << time[0] << std::endl;
#endif

// Clear odometry
/*if (no_data == true) {
    x.clear();
    y.clear();
    theta.clear();
    kappa.clear();
    got_pred = false;
}*/
       
// Update target odometry
x.push_back(in_x[0]);
y.push_back(in_y[0]);
theta.push_back(in_theta[0]);
kappa.push_back(in_kappa[0]);

got_pred = false;
    
if (x.size() >= units) {
    got_pred = true;
    std::fill_n(in_nn, UNITS, 0);
    std::fill_n(out_nn, CLASSES, 0);
    
    // Collect data
    auto it = x.begin();
    auto it0 = x.begin();
    for (int i=0; i<units; ++i, it++)
        in_nn[i] = *it - *it0;
    it = y.begin();
    it0 = y.begin();
    for (int i=units; i<units*2; ++i, it++)
        in_nn[i] = *it - *it0;
    it = theta.begin();
    for (int i=units*2; i<units*3; ++i, it++)
        in_nn[i] = *it;
    it = kappa.begin();
    for (int i=units*3; i<units*4; ++i, it++)
        in_nn[i] = *it;
    
    #ifdef DEBUG
    std::cout << "imported inputs:" << std::endl; 
    for (int i=0; i<UNITS; i++) {
        std::cout << in_nn[i] << " ";
    }
    std::cout << std::endl;
    #endif
    
    // Predict classes
    Som<SOM_ROWS, SOM_COLS, UNITS, CLASSES>(in_nn, out_nn);
}

// Outputs
for (int i=0; i<CLASSES; i++) {
    if (got_pred==true)
        out_conf[i] = out_nn[i];
    else
        out_conf[i] = 0.0;
}
out_class[0] = std::distance(out_conf, std::max_element(out_conf, out_conf + CLASSES))+1;
out_flag[0] = got_pred;

#ifdef DEBUG
std::cout << "pred flag: " << got_pred << std::endl;
std::cout << "outputs: "; 
for (int i=0; i<CLASSES; i++) {
    std::cout << out_conf[i] << " ";
}
std::cout << std::endl;
#endif
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_END --- EDIT HERE TO _BEGIN */
}

/*
 * Terminate function
 *
 */
void classifier_Terminate_wrapper(void)
{
/* %%%-SFUNWIZ_wrapper_Terminate_Changes_BEGIN --- EDIT HERE TO _END */
got_pred = false;
x.clear();
y.clear();
theta.clear();
kappa.clear();
/* %%%-SFUNWIZ_wrapper_Terminate_Changes_END --- EDIT HERE TO _BEGIN */
}

