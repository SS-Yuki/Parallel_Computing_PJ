#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <malloc.h>

#define MEASURE_VOID(__func, ...)                 \
    (((start = omp_get_wtime()), \
      __func(__VA_ARGS__),                        \
      (end = omp_get_wtime())),  \
     end - start)

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) >= 0.0) ? (x) : -(x))

#define PI 3.14159265358979323846

void RandomFill(double *d, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        d[i] = (double)rand() / RAND_MAX;
        d[i] = 2 * d[i] - 1.0;
    }
}

void naive_stencil(double *u, double alpha, int T, int N, double *u_ans, int lda) {
    for (int t = 1; t <= T; ++t) {
        for (int x = 1; x <= N; ++x) {
            for (int y = 1; y <= N; ++y) {
                if (t % 2 == 1) {
                    u_ans[x * lda + y] = alpha * (u[(x + 1) * lda + y] - 2 * u[x * lda + y] + u[(x - 1) * lda + y]) \
                                       + alpha * (u[x * lda + (y + 1)] - 2 * u[x * lda + y] + u[x * lda + (y - 1)]) \
                                       + u[x * lda + y];
                } else {
                    u[x * lda + y] = alpha * (u_ans[(x + 1) * lda + y] - 2 * u_ans[x * lda + y] + u_ans[(x - 1) * lda + y]) \
                                   + alpha * (u_ans[x * lda + (y + 1)] - 2 * u_ans[x * lda + y] + u_ans[x * lda + (y - 1)]) \
                                   + u_ans[x * lda + y];
                }
            }
        }
    }
    if (T % 2 == 0) {
        memcpy(u_ans, u, sizeof(double) * lda * lda);
    }
}

void student_stencil(double *u, double alpha, int T, int N, double *u_ans) {
    // naive_stencil(u, alpha, T, N, u_ans, N + 2);
    int lda = N + 2;
    for (int t = 1; t <= T; ++t) {
        #pragma omp parallel for collapse(2)
        for (int x = 1; x <= N; ++x) {
            for (int y = 1; y <= N; ++y) {
                int cur = x * lda + y;
                if (t % 2 == 1) {
                    u_ans[cur] = alpha * (u[(x + 1) * lda + y] - 2 * u[cur] + u[(x - 1) * lda + y]) \
                                       + alpha * (u[x * lda + (y + 1)] - 2 * u[cur] + u[x * lda + (y - 1)]) \
                                       + u[cur];
                } else {
                    u[cur] = alpha * (u_ans[(x + 1) * lda + y] - 2 * u_ans[cur] + u_ans[(x - 1) * lda + y]) \
                                   + alpha * (u_ans[x * lda + (y + 1)] - 2 * u_ans[cur] + u_ans[x * lda + (y - 1)]) \
                                   + u_ans[cur];
                }
            }
        }
    }
    if (T % 2 == 0) {
        memcpy(u_ans, u, sizeof(double) * lda * lda);
    }
}

void stencil_test(int T, int N) {
    // alloc memory
    double *u = (double*)malloc(sizeof(double) * (N + 2) * (N + 2));
    double *u_ans = (double*)malloc(sizeof(double) * (N + 2) * (N + 2));
    double *base = (double*)malloc(sizeof(double) * (N + 2) * (N + 2));
    double *base_ans = (double*)malloc(sizeof(double) * (N + 2) * (N + 2));    
    
    // initialize randomly
    srand(time(NULL));
    double alpha = (double)rand() / RAND_MAX * 1e-10;
    memset(u, 0, sizeof(double) * (N + 2) * (N + 2));
    memset(u_ans, 0, sizeof(double) * (N + 2) * (N + 2));
    memset(base_ans, 0, sizeof(double) * (N + 2) * (N + 2));
    for (int i = 1; i <= N; ++i) {
        RandomFill(u + i * (N + 2) + 1, N);
    }
    memcpy(base, u, sizeof(double) * (N + 2) * (N + 2));

    // test performance
    const int TRIAL = 5;
    double start, end;
    double t_min = __DBL_MAX__;
    for (int i = 0; i < TRIAL; i++) {
        memcpy(u, base, sizeof(double) * (N + 2) * (N + 2));
        double t = MEASURE_VOID(student_stencil, u, alpha, T, N, u_ans);
        t_min = MIN(t, t_min);
    }
    printf("minimal time spent: %.4f ms\n", t_min * 1000);
    fflush(stdout);

    // test correctness
    naive_stencil(base, alpha, T, N, base_ans, N + 2);
    double max_err = __DBL_MIN__;
    for (size_t i = 0; i < (N + 2) * (N + 2); i++) {
        double err = ABS(base_ans[i] - u_ans[i]);
        max_err = MAX(max_err, err);
    }
    const double threshold = 1e-9;
    const char * judge_s = (max_err < threshold) ? "correct" : "wrong!!!";
    printf("result: %s (err = %e)\n", judge_s, max_err);
    fflush(stdout);

    free(u);
    free(u_ans);
    free(base);
    free(base_ans);
}

int main(int argc, const char * argv[]) {
    if (argc != 3) {
        printf("Test usage: ./test T N\n");
        exit(-1);
    }

    int T = atoi(argv[1]);
    int N = atoi(argv[2]);
    if (N > 10000 || T > 10000) {
        printf("Input size is too big!\n");
        exit(-1);
    }

    printf("size: T: %d, N: %d\n", T, N);
    fflush(stdout);

    stencil_test(T, N);

    return 0;
}
