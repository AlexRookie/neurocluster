// Alessandro Antonucci @AlexRookie
// University of Trento

#ifndef RESHAPING_H
#define RESHAPING_H

// Keras reshaping layers

template<int Nrows, int Ncols>
void flatten(double in[Nrows][Ncols], double out[Nrows*Ncols]) {
    int s=0;
    for (int i=0; i<Nrows; i++) {
       for (int j=0; j<Ncols; j++) {
            out[s] = in[i][j];
            s++;
        }
    }
}

template<int Nrows, int Ncols>
void permute(double in[Nrows][Ncols], double out[Ncols][Nrows]) {
    for (int i=0; i<Ncols; i++) {
       for (int j=0; j<Nrows; j++) {
            out[i][j] = in[j][i];
        }
    }
}

#endif // RESHAPING_H