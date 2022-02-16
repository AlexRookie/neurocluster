// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

// Keras convolution layers

template <int Nin, int Nfilters, int Nkernel>
void conv1d(double in[Nin], const double kernel[1][Nkernel], const double b[Nfilters], double out[Nin-Nkernel+1][Nfilters]) {
    for (int f=0; f<Nfilters; f++) {
        for (int i=0; i<Nin-Nkernel+1; i++) {
            out[i][f] = 0.0;
        }
    }

    for (int f=0; f<Nfilters; f++) {
        for (int i=0; i<Nin-Nkernel+1; i++) {
            for (int j=0; j<Nkernel; j++) {
                out[i] += in[i+j] * kernel[0][j];
            }
            out[i] += b[i];
        }
    }
}

template <int Nrows, int Ncols, int Nfilters, int Nkernel>
void conv1d(double in[Nrows][Ncols], const double kernel[Nfilters][Ncols][Nkernel], const double b[Nfilters], double out[Nrows-Nkernel+1][Nfilters]) {
    for (int f=0; f<Nfilters; f++) {
        for (int i=0; i<Nrows-Nkernel+1; i++) {
            out[i][f] = 0.0;
        }
    }

    for (int f=0; f<Nfilters; f++) {
        for (int i=0; i<Nrows-Nkernel+1; i++) {
            for (int k=0; k<Nkernel; k++) {
                for (int j=0; j<Ncols; j++) {
                    out[i][f] += in[i+k][j] * kernel[f][j][k];
                }
            }
            out[i][f] += b[f];
        }
    }
}

#endif // CONVOLUTION_H