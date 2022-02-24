// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef RECURRENT_H
#define RECURRENT_H

// Keras recurrent layers

template <int Nin, int Nout, int steps>
void rnn(double in[steps][Nin], const double kernel[Nout][Nout], const double rec_kernel[Nout][Nin], double out[steps][Nout], double init[Nout]=NULL) {
    static double state[Nout] = {0.};
    if (init!=NULL) {
        for (int i=0; i<Nout; i++) {
            state[i] = init[i];
        }
    }
    for (int k=0; k<steps; k++) {
        double h[Nout];
        for (int j=0; j<Nout; j++) {
            h[j] = 0;
            for (int i=0; i<Nin; i++) {
                h[j] += in[k][i]*rec_kernel[j][i];
            }
        }
        for (int j=0; j<Nout; j++) {
            out[k][j] = 0;
            for (int i=0; i<Nout; i++) {
                out[k][j] += state[i]*kernel[j][i];
            }
            out[k][j] += h[j];
        }
        for (int i=0; i<Nout; i++) {
            state[i] = out[k][i];
        }
    }
}

#endif // RECURRENT_H