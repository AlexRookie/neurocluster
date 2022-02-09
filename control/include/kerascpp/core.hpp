// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef CORE_H
#define CORE_H

// Keras core layers

template <int Nin, int Nout>
void dense(double in[Nin], const double w[Nout][Nin], const double b[Nout], double out[Nout]) {
    for (int j=0; j<Nout; j++) {
        out[j] = 0;
        for (int i=0; i<Nin; i++) {
            out[j] += in[i]*w[j][i];
        }
        out[j] += b[j];
    }
}

template <int Nin, int Nker>
void conv1d(double in[Nin], const double kernel[1][Nker], double out[Nin-Nker+1]) {
    for (int i=0; i<Nin-Nker+1; i++) {
        out[i] = 0.0;
        for (int j=0; j<Nker; j++) {
            out[i] += in[i+j] * kernel[0][j];
        }
    }
}

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

#endif // CORE_H