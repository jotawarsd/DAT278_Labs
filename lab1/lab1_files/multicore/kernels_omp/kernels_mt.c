#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void triad(float *a, float *b, float *c, float *d, int n, int *active_threads) {
    omp_set_dynamic(0);
    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);

    #pragma omp parallel
    {
        #pragma omp single nowait
        *active_threads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 0; i < n / 8; i++) {
            int j = 8 * i;
            a[j]   = b[j]   + c[j]   * d[j];
            a[j+1] = b[j+1] + c[j+1] * d[j+1];
            a[j+2] = b[j+2] + c[j+2] * d[j+2];
            a[j+3] = b[j+3] + c[j+3] * d[j+3];
            a[j+4] = b[j+4] + c[j+4] * d[j+4];
            a[j+5] = b[j+5] + c[j+5] * d[j+5];
            a[j+6] = b[j+6] + c[j+6] * d[j+6];
            a[j+7] = b[j+7] + c[j+7] * d[j+7];
        }
    }
    for (int i = n - n % 8; i < n; i++)
        a[i] = b[i] + c[i] * d[i];
}


float dot_product(float *a, float *b, int n, int *active_threads) {
    omp_set_dynamic(0);
    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);

    float sum0=0.0f, sum1=0.0f, sum2=0.0f, sum3=0.0f;
    float sum4=0.0f, sum5=0.0f, sum6=0.0f, sum7=0.0f;
    #pragma omp parallel
    {
        #pragma omp single nowait
        *active_threads = omp_get_num_threads();
        #pragma omp for reduction(+:sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7) schedule(static)
        for (int i = 0; i < n / 8; i++) {
            int j = 8 * i;
            sum0 += a[j] * b[j]; sum1 += a[j+1] * b[j+1];
            sum2 += a[j+2] * b[j+2]; sum3 += a[j+3] * b[j+3];
            sum4 += a[j+4] * b[j+4]; sum5 += a[j+5] * b[j+5];
            sum6 += a[j+6] * b[j+6]; sum7 += a[j+7] * b[j+7];
        }
    }
    for (int i = n - n % 8; i < n; i++) sum0 += a[i] * b[i];
    return sum0+sum1+sum2+sum3+sum4+sum5+sum6+sum7;
}


void fma_chain(float *a, float *b, float *c, int n, int iters, int *active_threads) {
    omp_set_dynamic(0);
    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);

    #pragma omp parallel
    {
        #pragma omp single nowait
        *active_threads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 0; i < n / 8; i++) {
            int j = 8 * i;
            float x0=a[j], x1=a[j+1], x2=a[j+2], x3=a[j+3];
            float x4=a[j+4], x5=a[j+5], x6=a[j+6], x7=a[j+7];
            float y0=b[j], y1=b[j+1], y2=b[j+2], y3=b[j+3];
            float y4=b[j+4], y5=b[j+5], y6=b[j+6], y7=b[j+7];
            float c0=c[j], c1=c[j+1], c2=c[j+2], c3=c[j+3];
            float c4=c[j+4], c5=c[j+5], c6=c[j+6], c7=c[j+7];
            for (int k = 0; k < iters; k++) {
                x0=x0*y0+c0; x1=x1*y1+c1; x2=x2*y2+c2; x3=x3*y3+c3;
                x4=x4*y4+c4; x5=x5*y5+c5; x6=x6*y6+c6; x7=x7*y7+c7;
            }
            a[j]=x0; a[j+1]=x1; a[j+2]=x2; a[j+3]=x3;
            a[j+4]=x4; a[j+5]=x5; a[j+6]=x6; a[j+7]=x7;
        }
    }
    for (int i = n - n % 8; i < n; i++) {
        float x=a[i], y=b[i], ci=c[i];
        for (int k=0; k<iters; k++) x=x*y+ci;
        a[i]=x;
    }
}


void poly_eval(float *a, float *b, int n, int degree, int *active_threads) {
    omp_set_dynamic(0);
    int max_threads = omp_get_num_procs();
    omp_set_num_threads(max_threads);
    
    #pragma omp parallel
    {
        #pragma omp single nowait
        *active_threads = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 0; i < n / 8; i++) {
            int j = 8 * i;
            float x0=a[j], x1=a[j+1], x2=a[j+2], x3=a[j+3];
            float x4=a[j+4], x5=a[j+5], x6=a[j+6], x7=a[j+7];
            float y0=b[degree], y1=b[degree], y2=b[degree], y3=b[degree];
            float y4=b[degree], y5=b[degree], y6=b[degree], y7=b[degree];
            for (int d = degree - 1; d >= 0; d--) {
                y0=y0*x0+b[d]; y1=y1*x1+b[d]; y2=y2*x2+b[d]; y3=y3*x3+b[d];
                y4=y4*x4+b[d]; y5=y5*x5+b[d]; y6=y6*x6+b[d]; y7=y7*x7+b[d];
            }
            a[j]=y0; a[j+1]=y1; a[j+2]=y2; a[j+3]=y3;
            a[j+4]=y4; a[j+5]=y5; a[j+6]=y6; a[j+7]=y7;
        }
    }
    for (int i = n - n % 8; i < n; i++) {
        float x=a[i], y=b[degree];
        for (int d=degree-1; d>=0; d--) y=y*x+b[d];
        a[i]=y;
    }
}


#define FLUSH_SIZE (32 * 1024 * 1024) // 32 MB

void flush_cache() {
    static volatile char *flush_buffer = NULL;
    if (!flush_buffer) {
        flush_buffer = malloc(FLUSH_SIZE);
        if (!flush_buffer)
            return; // failed allocation
    }

    volatile char sink = 0;

    for (size_t i = 0; i < FLUSH_SIZE; i += 64) {
        flush_buffer[i] = (char)i;
        sink += flush_buffer[i];
    }

    (void)sink;
}


int main(int argc, char *argv[]) {

    const char *kernel = argv[1];
    int N = atoi(argv[2]);
    int param = (argc > 3) ? atoi(argv[3]) : 200; // iters for fma, degree for poly

    int active_threads = 1;

    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *c = (float *)malloc(N * sizeof(float));
    float *d = (float *)malloc(N * sizeof(float));

    if (!a || !b || !c || !d) {
        printf("Allocation failed.\n");
        free(a); free(b); free(c); free(d);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 3.0f;
        d[i] = 4.0f;
    }

    const int RUNS = 20;
    double total_time = 0.0;
    for (int run = 0; run < RUNS; run++) {
        flush_cache();
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        if (strcmp(kernel, "dot") == 0) {
            volatile float result = dot_product(a, b, N, &active_threads);
            (void)result;
        } else if (strcmp(kernel, "triad") == 0) {
            triad(a, b, c, d, N, &active_threads);
        } else if (strcmp(kernel, "fma") == 0) {
            fma_chain(a, b, c, N, param, &active_threads);
        } else if (strcmp(kernel, "poly") == 0) {
            poly_eval(a, b, N, param, &active_threads);
        } else {
            printf("Unknown kernel: %s\n", kernel);
            free(a); free(b); free(c); free(d);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        total_time += elapsed;
    }

    double avg_time = total_time / (double)RUNS;

    double flops = 0.0;
    if (strcmp(kernel, "dot") == 0)
        flops = 2.0 * N;
    else if (strcmp(kernel, "triad") == 0)
        flops = 2.0 * N;
    else if (strcmp(kernel, "fma") == 0)
        flops = 2.0 * N * param;
    else if (strcmp(kernel, "poly") == 0)
        flops = 2.0 * N * param;

    double gflops = flops / (avg_time * 1e9);

    FILE *fout = fopen("output_sample.txt", "w");
    if (fout) {
        for (int i = 0; i < 5 && i < N; i++) {
            fprintf(fout, "%f\n", a[i]);
        }
        fclose(fout);
    }

    printf("Kernel: %-5s | N: %d | Threads: %d | Avg Time: %.6f s | Avg GFLOPs: %.3f\n",
           kernel, N, active_threads, avg_time, gflops);

    free(a); free(b); free(c); free(d);
    return 0;
}
