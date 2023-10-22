#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef SERVER
#include <fftw3.h>
#endif

#define MEASURE_VOID(__func, ...)                                              \
  (((start = omp_get_wtime()), __func(__VA_ARGS__), (end = omp_get_wtime())),  \
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

void naive_dft(int n, const double complex *in, double complex *out) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      out[i] +=
          in[j] * (cos(-2 * PI * i * j / n) + sin(-2 * PI * i * j / n) * I);
    }
  }
}

void naive_fft(int n, const double complex *in, double complex *out) {
  if (n <= 1) {
    out[0] = in[0];
    return;
  }
  const double theta0 = 2 * PI / n;
  double complex *even = malloc(n / 2 * sizeof(double complex));
  double complex *odd = malloc(n / 2 * sizeof(double complex));

  for (int i = 0; i < n / 2; i++) {
    even[i] = in[2 * i];
    odd[i] = in[2 * i + 1];
  }

  // recursive
  naive_fft(n / 2, even, even);
  naive_fft(n / 2, odd, odd);

  for (int i = 0; i < n / 2; i++) {
    double complex theta_i = theta0 * i;
    out[i] = even[i] + odd[i] * (cos(theta_i) - sin(theta_i) * I);
    out[i + n / 2] = even[i] - odd[i] * (cos(theta_i) - sin(theta_i) * I);
  }
  free(even);
  free(odd);
}

void transpose(int M, int N, const double complex *A, double complex *B, int m,
               int n, int i, int j) {
  if (m == 1 && n == 1) {
    B[j * M + i] = A[i * N + j];
    return;
  }
  if (n >= m) {
    int n_new = n / 2;
    transpose(M, N, A, B, m, n_new, i, j);
    transpose(M, N, A, B, m, n - n_new, i, j + n_new);
  } else {
    int m_new = m / 2;
    transpose(M, N, A, B, m_new, n, i, j);
    transpose(M, N, A, B, m - m_new, n, i + m_new, j);
  }
}

void student_fft(int n, const double complex *in, double complex *out) {
  omp_set_num_threads(24);
  const double theta0 = 2 * PI / n;

  if (n == 1) {
    out[0] = in[0];
    return;
  }
  if (n == 2) {
    out[0] = in[0] + in[1];
    out[1] = in[0] - in[1];
    return;
  }

  double complex *arr1 = (double complex *)malloc(sizeof(double complex) * n);
  double complex *arr2 = (double complex *)malloc(sizeof(double complex) * n);

  // 求出n1, n2
  int n1, n2;
  int ret = 0;
  int tmp = n;
  while (tmp >>= 1) {
    ret += 1;
  }
  n2 = 1 << (ret / 2);
  n1 = n / n2;

  // x: n1xn2 (j1, j2)    y: n2xn1 (i2, i1)         (n1==n2 or n1==2*n2)

  // step1: 转置
  transpose(n1, n2, in, arr1, n1, n2, 0, 0);

  // step2: FFT
#pragma omp parallel for
  for (int i = 0; i < n2; i++) {
    naive_fft(n1, arr1 + n1 * i, arr2 + n1 * i);
  }

// step3: 乘
#pragma omp parallel for
  for (int i = 0; i < n2; i++) {
    double theta_i = theta0 * i;
    for (int j = 0; j < n1; j++) {
      arr2[i * n1 + j] *= (cos(theta_i * j) - sin(theta_i * j) * I);
    }
  }

  // step4: 转置
  transpose(n2, n1, arr2, arr1, n2, n1, 0, 0);

  // step5:FFT
#pragma omp parallel for
  for (int i = 0; i < n1; i++) {
    naive_fft(n2, arr1 + n2 * i, arr2 + n2 * i);
  }

  // step6: 转置
  transpose(n1, n2, arr2, out, n1, n2, 0, 0);

  free(arr1);
  free(arr2);
}

void fft_test(int n) {
  // alloc memory
  double complex *in, *out, *ans;
  in = (double complex *)malloc(sizeof(double complex) * n);
  out = (double complex *)malloc(sizeof(double complex) * n);
  ans = (double complex *)malloc(sizeof(double complex) * n);

  // initialize randomly
  srand(time(NULL));
  RandomFill((double *)in, 2 * n);

  // test performance
  const int TRIAL = 5;
  double start, end;
  double t_min = __DBL_MAX__;
  for (int i = 0; i < TRIAL; i++) {
    memset(out, 0, sizeof(double complex) * n);
    double t = MEASURE_VOID(student_fft, n, in, out);
    t_min = MIN(t, t_min);
  }

  printf("minimal time spent: %.4f ms\n", t_min * 1000);
  fflush(stdout);

// baseline
#ifdef SERVER
  t_min = __DBL_MAX__;
  fftw_plan p;
  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
  p = fftw_plan_dft_1d(n, in, ans, FFTW_FORWARD, FFTW_ESTIMATE);
  for (int i = 0; i < TRIAL; ++i) {
    double t = MEASURE_VOID(fftw_execute, p);
    t_min = MIN(t, t_min);
  }
  printf("threads: %d, baseline time spent: %.4f ms\n", fftw_planner_nthreads(),
         t_min * 1000);
  fflush(stdout);
#else
  naive_fft(n, in, ans);
#endif

  // test correctness
  double max_err = __DBL_MIN__;
  for (size_t i = 0; i < n; i++) {
    double err = cabs(ans[i] - out[i]);
    max_err = MAX(max_err, err);
  }
  const double threshold = 1e-7;
  const char *judge_s = (max_err < threshold) ? "correct" : "wrong!!!";
  printf("result: %s (err = %e)\n", judge_s, max_err);
  fflush(stdout);

  free(in);
  free(out);
  free(ans);
}

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    printf("Test usage: ./test N\n");
    exit(-1);
  }

  int n = atoi(argv[1]);
  if (n > 30) {
    printf("Input size is too big!\n");
    exit(-1);
  }
  n = (1 << n);

  printf("size: %d\n", n);
  fflush(stdout);

  fft_test(n);

  return 0;
}
