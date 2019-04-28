//
//  supporting.h
//  Stochastic Approximation
//
//  Created by jjxu on 2019/3/31.
//  Copyright © 2019 jjxu. All rights reserved.
//



#include <stdio.h>

#define        BLOCKSIZE            256

typedef struct{
    int        NUM_REPS;            /* Maximum number of replications that can be carried out. */
    long long *RUN_SEED;        /* seed used during optimization */
    double     TOLERANCE;             /* Tolerance level used to differentiate objective value improvemen*/
    int        N1;            /* total number of iteration  */
    int        N2;            /* number of samples for one subgradient estimation */
    double        a;        /* the constant for diminishing step size */
    long long  EVAL_SEED;
}configType;

int solveSub(double *x, double *lambda, long long *SEED);
FILE *open_ofile(char *name);

int readConfig();
double get_CI(double *CI, double* x);
float scalit(float lower, float upper, long long *RUN_SEED);
float randUniform(long long *SEED);
