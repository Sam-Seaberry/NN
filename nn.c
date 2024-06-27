#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

typedef struct
{
    Mat a0, a1, a2;
    Mat w1, b1;
    Mat w2, b2;
}Xor;

Xor xor_alloc(void){
    Xor or;
    
    or.a0 = mat_alloc(1,2);
    or.w1 = mat_alloc(2,2);
    or.b1 = mat_alloc(1,2);
    or.a1 = mat_alloc(1,2);

    or.w2 = mat_alloc(2,1);
    or.b2 = mat_alloc(1,1);
    or.a2 = mat_alloc(1,1);

    return or;
}



void forward_xor(Xor m){

    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);

}

float cost(Xor m, Mat ti, Mat to){
    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);
    size_t n =ti.rows;

    float c = 0;
    for(size_t i = 0; i < n; ++i){
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m.a0, x);
        forward_xor(m);

        size_t q = to.cols;

        for(size_t j = 0; j< q; ++j){
            float d = MAT_AT(m.a2, 0, j) - MAT_AT(y,0,j);
            c += d*d;
        }
    }
    return c/n;
}

void learn(Xor m, Xor g, float rate){
    for(size_t i =0; i< m.w1.rows; ++i){
        for(size_t j =0; j< m.w1.cols; ++j){
            MAT_AT(m.w1, i, j) -= rate*MAT_AT(g.w1, i, j);
        }
    }
    for(size_t i =0; i< m.b1.rows; ++i){
        for(size_t j =0; j< m.b1.cols; ++j){
            MAT_AT(m.b1, i, j) -= rate*MAT_AT(g.b1, i, j);
        }
    }
    for(size_t i =0; i< m.w2.rows; ++i){
        for(size_t j =0; j< m.w2.cols; ++j){
            MAT_AT(m.w2, i, j) -= rate*MAT_AT(g.w2, i, j);
        }
    }
    for(size_t i =0; i< m.b2.rows; ++i){
        for(size_t j =0; j< m.b2.cols; ++j){
            MAT_AT(m.b2, i, j) -= rate*MAT_AT(g.b2, i, j);
        }
    }
}

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to){
    float saved;
    float c = cost(m, ti, to);
    for(size_t i =0; i< m.w1.rows; ++i){
        for(size_t j =0; j< m.w1.cols; ++j){
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }
    for(size_t i =0; i< m.b1.rows; ++i){
        for(size_t j =0; j< m.b1.cols; ++j){
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.b1, i, j) = saved;
        }
    }
    for(size_t i =0; i< m.w2.rows; ++i){
        for(size_t j =0; j< m.w2.cols; ++j){
            saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.w2, i, j) = saved;
        }
    }
    for(size_t i =0; i< m.b2.rows; ++i){
        for(size_t j =0; j< m.b2.cols; ++j){
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.b2, i, j) = saved;
        }
    }
}

int main(void){
    
    srand(time(0));

    size_t stride = 3;
    size_t n = (sizeof(td)/sizeof(td[0]))/stride;

    float eps = 1e-2;
    float rate = 1e-2;
    
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2
    };

    size_t arch[] = {2,2,1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);
    nn_rand(g, 0,1);

    //mat_copy(NN_INPUT(nn),mat_row(ti,1));
    //nn_forward(nn);
    //MAT_PRINT(NN_OUTPUT(nn));

    printf("cost= %f\n", nn_cost(nn, ti,to));
    for (size_t i = 0; i< 100000; ++i){
        nn_fdiff(nn, g, eps, ti, to);
        nn_learn(nn, g, rate);
    }
    printf("cost= %f\n", nn_cost(nn, ti,to));

    for (size_t i =0; i<2; ++i){
        for(size_t j =0;j<2;++j){
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;
            nn_forward(nn);
            float y = *NN_OUTPUT(nn).es;
            printf("%zu ^ %zu = %f\n",i,j,y);
            
        }
    }

    return 0;
    /*
    Xor or = xor_alloc();
    Xor g = xor_alloc();


    mat_rand(or.w1, 0, 1);
    mat_rand(or.b1, 0, 1);
    mat_rand(or.w2, 0, 1);
    mat_rand(or.b2, 0, 1);
    

    float eps = 1e-1;
    float rate = 1e-1;

    printf("cost = %f\n", cost(or,ti, to));
    for(size_t i =0; i<100000; ++i){
        finite_diff(or, g, eps,ti,to);
        learn(or ,g, rate);
        //printf("%zu: cost = %f\n", i, cost(or,ti, to));
    }
    printf("cost = %f\n", cost(or,ti, to));

    
  
    for (size_t i =0; i<2; ++i){
        for(size_t j =0;j<2;++j){
            MAT_AT(or.a0, 0, 0) = i;
            MAT_AT(or.a0, 0, 1) = j;
            forward_xor(or);
            float y = *or.a2.es;
            printf("%zu ^ %zu = %f\n",i,j,y);
            
        }
    }

    
    return 0; */
}