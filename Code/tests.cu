#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cstdint>
#include <random>
#include <iostream>

#include "qu.cu"


//#include "dsingle_functions.cu"
#include "error_bound_dsingle.cu"
#include "kernels.cu"
#define NUM_BLOCKS (1000)
#define THREADS_PER_BLOCK (1024)

#define N (THREADS_PER_BLOCK*NUM_BLOCKS)
// element op element

//Host code
template<typename T, typename U>
void test_addition(T const& x , U const& y){
  printf("Testing extended_add\n");
  dsingle a, b, c;
  a  = dsingle(x);
  b =  dsingle(y);
  
  dsingle *da, *db, *dc;
  cudaMalloc((void **)&da,sizeof(dsingle));
  cudaMalloc((void **)&db,sizeof(dsingle));
  cudaMalloc((void **)&dc,sizeof(dsingle));

  cudaMemcpy(da, &a, sizeof(dsingle),cudaMemcpyHostToDevice);
  cudaMemcpy(db, &b, sizeof(dsingle),cudaMemcpyHostToDevice);
  addCUDA<<<30,32>>>(da,db,dc);
  cudaMemcpy(&c, dc, sizeof(dsingle),cudaMemcpyDeviceToHost);

  double truev = x + y;
  printf("%.16f : native double\n",truev);
  printf("%.16f : extended\n", c.evaluate());
  printf("%.16f : diffrence\n", c.evaluate() - truev);
  printf("---\n");

  cudaFree( da );
  cudaFree( db );
  cudaFree( dc );
}

template<typename T, typename U>
void test_subtraction(T const& x , U const& y){
  printf("Testing extended_subtract\n");
  dsingle a, b, c;
  a  = dsingle(x);
  b =  dsingle(y);
  
  dsingle *da, *db, *dc;
  cudaMalloc((void **)&da,sizeof(dsingle));
  cudaMalloc((void **)&db,sizeof(dsingle));
  cudaMalloc((void **)&dc,sizeof(dsingle));

  cudaMemcpy(da, &a, sizeof(dsingle),cudaMemcpyHostToDevice);
  cudaMemcpy(db, &b, sizeof(dsingle),cudaMemcpyHostToDevice);
  subCUDA<<<30,32>>>(da,db,dc);
  cudaMemcpy(&c, dc, sizeof(dsingle),cudaMemcpyDeviceToHost);


  double truev = x  -  y;
  printf("%.16f : native double\n",truev);
  printf("%.16f : extended\n", c.evaluate());
  printf("%.16f : diffrence\n", c.evaluate() - truev);
  printf("---\n");

  cudaFree( da );
  cudaFree( db );
  cudaFree( dc );
}

template<typename T, typename U>
void test_multiplcation(T const& x , U const& y){
  printf("Testing extended_multiply\n");
  dsingle a, b, c;
  a  = dsingle(x);
  b =  dsingle(y);
  
  dsingle *da, *db, *dc;
  cudaMalloc((void **)&da,sizeof(dsingle));
  cudaMalloc((void **)&db,sizeof(dsingle));
  cudaMalloc((void **)&dc,sizeof(dsingle));

  cudaMemcpy(da, &a, sizeof(dsingle),cudaMemcpyHostToDevice);
  cudaMemcpy(db, &b, sizeof(dsingle),cudaMemcpyHostToDevice);
  mulCUDA<<<30,32>>>(da,db,dc);
  cudaMemcpy(&c, dc, sizeof(dsingle),cudaMemcpyDeviceToHost);

  double truev = x *  y;
  printf("%.16f : native double\n",truev);
  printf("%.16f : extended\n", c.evaluate());
  printf("%.16f : diffrence\n", c.evaluate() - truev);
  printf("---\n");

  cudaFree( da );
  cudaFree( db );
  cudaFree( dc );
}

template<typename T, typename U>
void test_division(T const& x , U const& y){
  printf("Testing extended_divide\n");
  dsingle a, b, c;
  a  = dsingle(x);
  b =  dsingle(y);
  
  dsingle *da, *db, *dc;
  cudaMalloc((void **)&da,sizeof(dsingle));
  cudaMalloc((void **)&db,sizeof(dsingle));
  cudaMalloc((void **)&dc,sizeof(dsingle));

  cudaMemcpy(da, &a, sizeof(dsingle),cudaMemcpyHostToDevice);
  cudaMemcpy(db, &b, sizeof(dsingle),cudaMemcpyHostToDevice);
  divCUDA<<<30,32>>>(da,db,dc);
  cudaMemcpy(&c, dc, sizeof(dsingle),cudaMemcpyDeviceToHost);

  double truev = x /  y;
  printf("%.16f : native double\n",truev);
  printf("%.16f : extended\n", c.evaluate());
  printf("%.16f : diffrence\n", c.evaluate()- truev);
  printf("---\n");

  cudaFree( da );
  cudaFree( db );
  cudaFree( dc );
}



int test_extended_vector(){
  dsingle *a, *b, *c;
  dsingle *d_a, *d_b, *d_c;
  int size = N * sizeof( dsingle );
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* allocate space for device copies of a, b, c */

  cudaMalloc( (void **) &d_a, size );
  cudaMalloc( (void **) &d_b, size );
  cudaMalloc( (void **) &d_c, size );

  /* allocate space for host copies of a, b, c and setup input values */

  a = (dsingle *)malloc( size );
  b = (dsingle *)malloc( size );
  c = (dsingle *)malloc( size );

  for( int i = 0; i < N; i++ )
  {
    a[i] = b[i] = dsingle((double)i);
    c[i] =dsingle(0.0);
  }

  /* copy inputs to device */
  /* fix the parameters needed to copy data to the device */
  cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

  /* launch the kernel on the GPU */
  /* insert the launch parameters to launch the kernel properly using blocks and threads */ 
  cudaEventRecord(start);
  mulVecCUDA<<< (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c);
  cudaEventRecord(stop);
  //cudaDeviceSynchronize();
  /* copy result back to host */
  /* fix the parameters needed to copy data back to the host */
  cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf( "c[%d] = %.14e\n",N-1, (double) c[N-1].evaluate());
  printf("Time for extended: %.16f\n", milliseconds);

  /* clean up */

  free(a);
  free(b);
  free(c);
  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );
  
  return 0;
}
int test_vector(){
  double *a, *b, *c;
  double *d_a, *d_b, *d_c;
  int size = N * sizeof( double );
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* allocate space for device copies of a, b, c */

  cudaMalloc( (void **) &d_a, size );
  cudaMalloc( (void **) &d_b, size );
  cudaMalloc( (void **) &d_c, size );

  /* allocate space for host copies of a, b, c and setup input values */

  a = (double *)malloc( size );
  b = (double *)malloc( size );
  c = (double *)malloc( size );

  for( int i = 0; i < N; i++ )
  {
    a[i] = b[i] = i;
    c[i] =0.0;
  }
  /* copy inputs to device */
  /* fix the parameters needed to copy data to the device */
  cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

  /* launch the kernel on the GPU */
  /* insert the launch parameters to launch the kernel properly using blocks and threads */ 
  cudaEventRecord(start);
  mulVecCUDA<<< (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c);
  cudaEventRecord(stop);
  //cudaDeviceSynchronize();
  /* copy result back to host */
  /* fix the parameters needed to copy data back to the host */
  cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);


  printf( "c[%d] = %.16e\n",N-1, c[N-1]);
  printf("Time for native: %.16f\n", milliseconds);

  /* clean up */

  free(a);
  free(b);
  free(c);
  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );
  
  return 0;
}
int test_float_vector(){
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  int size = N * sizeof( float );
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* allocate space for device copies of a, b, c */

  cudaMalloc( (void **) &d_a, size );
  cudaMalloc( (void **) &d_b, size );
  cudaMalloc( (void **) &d_c, size );

  /* allocate space for host copies of a, b, c and setup input values */

  a = (float *)malloc( size );
  b = (float *)malloc( size );
  c = (float *)malloc( size );

  for( int i = 0; i < N; i++ )
  {
    a[i] = b[i] = i;
    c[i] =0.0;
  }
  /* copy inputs to device */
  /* fix the parameters needed to copy data to the device */
  cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

  /* launch the kernel on the GPU */
  /* insert the launch parameters to launch the kernel properly using blocks and threads */ 
  cudaEventRecord(start);
  mulVecCUDA<<< (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c);
  cudaEventRecord(stop);
  //cudaDeviceSynchronize();
  /* copy result back to host */
  /* fix the parameters needed to copy data back to the host */
  cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);


  printf( "c[%d] = %.16e\n",N-1, c[N-1]);
  printf("Time for floats: %.16f\n", milliseconds);

  /* clean up */

  free(a);
  free(b);
  free(c);
  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );
  
  return 0;
}

template<class T>
T test_pi(int itr,
          float &milliseconds,
          int repeats){
  T result, *d_result;
  cudaMalloc((void **)&d_result,sizeof(T));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < repeats; ++i){
    pi<<<1,1>>>(d_result, itr);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(&result, d_result, sizeof(d_result),cudaMemcpyDeviceToHost);
  cudaFree( d_result );
  return result;
}

template<class T,class U>
T* test_vector(double *vectorA, 
               double *vectorB, 
               void operation (T *a, U *b, T *c),
               float &milliseconds,
               int repeats) {
  T *d_a, *d_c, *a, *c;
  U *d_b, *b;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* allocate space for device copies of a, b, c */

  cudaMalloc( (void **) &d_a, N * sizeof( T ) );
  cudaMalloc( (void **) &d_b, N * sizeof( U ) );
  cudaMalloc( (void **) &d_c, N * sizeof( T ) );

  /* allocate space for host copies of a, b, c and setup input values */

  a = (T *)malloc( N * sizeof( T ) );
  b = (U *)malloc(N * sizeof( U ));
  c = (T *)malloc( N * sizeof( T ) );

  //convert to correct data types
  for( int i = 0; i < N; i++ ){
    a[i] = (T) vectorA[i];
    b[i] = (U) vectorB[i];
  }

  /* copy inputs to device */
  /* fix the parameters needed to copy data to the device */
  cudaMemcpy( d_a, a, N * sizeof( T ), cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, b, N * sizeof( U ), cudaMemcpyHostToDevice );

  /* launch the kernel on the GPU */
  /* insert the launch parameters to launch the kernel properly using blocks and threads */ 
  cudaEventRecord(start);
  for (int i = 0; i < repeats; ++i){
    operation<<< (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c);
  }
  

  cudaEventRecord(stop);
  //cudaDeviceSynchronize();
  /* copy result back to host */
  /* fix the parameters needed to copy data back to the host */
  cudaMemcpy( c, d_c, N * sizeof( T ), cudaMemcpyDeviceToHost );
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  //printf( "c[%d] = %.16e\n",N-1, c[N-1]);
  //printf("Time: %.16f\n", milliseconds);
  free(a);
  free(b);
  cudaFree( d_a );
  cudaFree( d_b );
  cudaFree( d_c );
  return c;
}
union Ud{
  Ud(double num = 0.0) : d(num) {}
  unsigned long long u;
  double d;
};

unsigned long long ULP(dsingle x, double y){
  Ud a(x.evaluate());
  Ud b(y);
  return (a.u > b.u) ? a.u - b.u : b.u - a.u;
}
unsigned long long ULP(float x, double y){
  Ud a((double)x);
  Ud b(y);
  return (a.u > b.u) ? a.u - b.u : b.u - a.u;
}
unsigned long long ULP(double x, double y){
  Ud a(x);
  Ud b(y);
  return (a.u > b.u) ? a.u - b.u : b.u - a.u;
}

template<typename T>
unsigned long long correctness(T *vectorA, double *vectorB){
  unsigned long long ret = 0;
  for( int i = 0; i < N; i++ ){
    ret += ULP(vectorA[i],vectorB[i]);
  }
  return ret/N;
}
unsigned long long *correctnessArray(float *vectorA, double *vectorB){
  unsigned long long *ret;
  ret = (unsigned long long *)malloc( N * sizeof( unsigned long long ) );
  for( int i = 0; i < N; i++ ){
    Ud a((double)vectorA[i]);
    Ud b(vectorB[i]);
    (a.u > b.u) ? ret[i] = a.u - b.u : ret[i] = b.u - a.u;
  }
  return ret;
}

unsigned long long *correctnessArray(double *vectorA, double *vectorB){
  unsigned long long *ret;
  ret = (unsigned long long *)malloc( N * sizeof( unsigned long long ) );
  for( int i = 0; i < N; i++ ){
    Ud a(vectorA[i]);
    Ud b(vectorB[i]);
    (a.u > b.u) ? ret[i] = a.u - b.u : ret[i] = b.u - a.u;
  }
  return ret;
}
unsigned long long *correctnessArray(dsingle *vectorA, double *vectorB){
  unsigned long long *ret;
  ret = (unsigned long long *)malloc( N * sizeof( unsigned long long ) );
  for( int i = 0; i < N; i++ ){
    Ud a(vectorA[i].evaluate());
    Ud b(vectorB[i]);
    (a.u > b.u) ? ret[i] = a.u - b.u : ret[i] = b.u - a.u;
  }
  return ret;
}

template<typename T, typename U>
void addVecCPU(T *a, U *b, T *c){
  printf("Starting CPU operation...");
   for( int i = 0; i < N; i++ ){
     c[i] = a[i] + b[i];
  }
  printf("Finished\n");
}
template<typename T, typename U>
void subVecCPU(T *a, U *b, T *c){
  printf("Starting CPU operation...");
   for( int i = 0; i < N; i++ ){
     c[i] = a[i] - b[i];
  }
  printf("Finished\n");
}
template<typename T, typename U>
void mulVecCPU(T *a, U *b, T *c){
  printf("Starting CPU operation...");
  for( int i = 0; i < N; i++ ){
     c[i] = a[i] * b[i];
  }
  printf("Finished\n");
}
template<typename T, typename U>
void divVecCPU(T *a, U *b, T *c){
  printf("Starting CPU operation...");
   for( int i = 0; i < N; i++ ){
     c[i] = a[i] / b[i];
  }
  printf("Finished\n");
}

int main(int argc, char const *argv[]){
  deviceQuery();





  //test_addition(1.1415926535897932384626433832795, 3.0);
  //test_subtraction(1.1415926535897932384626433832795, 3.0);
  //test_multiplcation(1.1415926535897932384626433832795, 3.0);
  //test_division(.1689876543, .935123456);
  // -std=c++11
  bool saveData = false;

  //create data
  double *a, *b, *c;
  int size = N * sizeof( double );
  a = (double *)malloc( size );
  b = (double *)malloc( size );
  c = (double *)malloc( size );

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution1(-1000000.0,1000000.0);
  std::uniform_real_distribution<double> distribution2(-1000.0,0.0);
  for( int i = 0; i < N; i++ ){
    a[i] = distribution1(generator);
    b[i] = distribution1(generator);
  }
  
  double *double_c;
  float *float_c;
  dsingle *dsingle_c;

  //Timers for performance testing
  float double_time = 0.0;
  float float_time = 0.0;
  float dsingle_time = 0.0;
  int repeats = 100000;

  // arrays to hold correctness tests
  unsigned long long *arr1, *arr2, *arr3;

  printf("---\nadd\n");
  addVecCPU(a,b,c);
  printf("Starting GPU operations...\n");

  double_c = test_vector<double,double>(a,b,addVecCUDA,double_time, repeats);
  printf("Time of double: %.16f\n",double_time/repeats);

  dsingle_c = test_vector<dsingle,dsingle>(a,b,addVecCUDA,dsingle_time, repeats);
  printf("Time of EP: %.16f\n",dsingle_time/repeats);

  float_c = test_vector<float,float>(a,b,addVecCUDA,float_time, repeats);
  printf("Time of float: %.16f\n",float_time/repeats);

  printf("ULP of double: %I64u\n",correctness(double_c, c));
  printf("ULP of EP: %I64u\n",correctness(dsingle_c, c));
  printf("ULP of float: %I64u\n",correctness(float_c, c));

  
  
  

  printf("Speed-up: %.16f\n", double_time/dsingle_time);

  arr1 = correctnessArray(double_c, c);
  arr2 = correctnessArray(dsingle_c, c);
  arr3 = correctnessArray(float_c, c);
  FILE *f;
  if(saveData){
    FILE *f = fopen("add.csv", "w");
    fprintf(f, "a,b,c,double_c,epHi,epLo,epEval,float_c,doubleULP,epULP,floatULP\n");
    for( int i = 0; i < N; i++ ){
      fprintf(f, "%.16e,", a[i]);
      fprintf(f, "%.16e,", b[i]);
      fprintf(f, "%.16e,", c[i]);
      fprintf(f, "%.16e,", double_c[i]);
      fprintf(f, "%.16e,", dsingle_c[i].hi());
      fprintf(f, "%.16e,", dsingle_c[i].lo());
      fprintf(f, "%.16e,", dsingle_c[i].evaluate());
      fprintf(f, "%.16e,", float_c[i]);
      fprintf(f, "%I64u,", arr1[i]);
      fprintf(f, "%I64u,", arr2[i]);
      fprintf(f, "%I64u\n", arr3[i]);
    }
    fclose(f);
  }
    free(arr1);
    free(arr2);
    free(arr3);


  printf("---\nsub\n");
  subVecCPU(a,b,c);
  printf("Starting GPU operations...\n");

  double_c = test_vector<double,double>(a,b,subVecCUDA,double_time, repeats);
  printf("Time of double: %.16f\n",double_time/repeats);

  dsingle_c = test_vector<dsingle,dsingle>(a,b,subVecCUDA,dsingle_time, repeats);
  printf("Time of EP: %.16f\n",dsingle_time/repeats);

  float_c = test_vector<float,float>(a,b,subVecCUDA,float_time, repeats);
  printf("Time of float: %.16f\n",float_time/repeats);

  printf("ULP of double: %I64u\n",correctness(double_c, c));
  printf("ULP of EP: %I64u\n",correctness(dsingle_c, c));
  printf("ULP of float: %I64u\n",correctness(float_c, c));

  
  
  

  printf("Speed-up: %.16f\n", double_time/dsingle_time);

  arr1 = correctnessArray(double_c, c);
  arr2 = correctnessArray(dsingle_c, c);
  arr3 = correctnessArray(float_c, c);
  if(saveData){
    f = fopen("sub.csv", "w");
    fprintf(f, "a,b,c,double_c,epHi,epLo,epEval,float_c,doubleULP,epULP,floatULP\n");
    for( int i = 0; i < N; i++ ){
      fprintf(f, "%.16e,", a[i]);
      fprintf(f, "%.16e,", b[i]);
      fprintf(f, "%.16e,", c[i]);
      fprintf(f, "%.16e,", double_c[i]);
      fprintf(f, "%.16e,", dsingle_c[i].hi());
      fprintf(f, "%.16e,", dsingle_c[i].lo());
      fprintf(f, "%.16e,", dsingle_c[i].evaluate());
      fprintf(f, "%.16e,", float_c[i]);
      fprintf(f, "%I64u,", arr1[i]);
      fprintf(f, "%I64u,", arr2[i]);
      fprintf(f, "%I64u\n", arr3[i]);
    }
    fclose(f);
  }
  free(arr1);
  free(arr2);
  free(arr3);
  

  printf("---\nmul\n");
  mulVecCPU(a,b,c);
  printf("Starting GPU operations...\n");

  double_c = test_vector<double,double>(a,b,mulVecCUDA,double_time, repeats);
  printf("Time of double: %.16f\n",double_time/repeats);

  dsingle_c = test_vector<dsingle,dsingle>(a,b,mulVecCUDA,dsingle_time, repeats);
  printf("Time of EP: %.16f\n",dsingle_time/repeats);

  float_c = test_vector<float,float>(a,b,mulVecCUDA,float_time, repeats);
  printf("Time of float: %.16f\n",float_time/repeats);

  printf("ULP of double: %I64u\n",correctness(double_c, c));
  printf("ULP of EP: %I64u\n",correctness(dsingle_c, c));
  printf("ULP of float: %I64u\n",correctness(float_c, c));

  
  
  

  printf("Speed-up: %.16f\n", double_time/dsingle_time);

  arr1 = correctnessArray(double_c, c);
  arr2 = correctnessArray(dsingle_c, c);
  arr3 = correctnessArray(float_c, c);
  if(saveData){
    f = fopen("mul.csv", "w");
    fprintf(f, "a,b,c,double_c,epHi,epLo,epEval,float_c,doubleULP,epULP,floatULP\n");
    for( int i = 0; i < N; i++ ){
      fprintf(f, "%.16e,", a[i]);
      fprintf(f, "%.16e,", b[i]);
      fprintf(f, "%.16e,", c[i]);
      fprintf(f, "%.16e,", double_c[i]);
      fprintf(f, "%.16e,", dsingle_c[i].hi());
      fprintf(f, "%.16e,", dsingle_c[i].lo());
      fprintf(f, "%.16e,", dsingle_c[i].evaluate());
      fprintf(f, "%.16e,", float_c[i]);
      fprintf(f, "%I64u,", arr1[i]);
      fprintf(f, "%I64u,", arr2[i]);
      fprintf(f, "%I64u\n", arr3[i]);
    }
    fclose(f);
  }
  free(arr1);
  free(arr2);
  free(arr3);

  printf("---\ndiv\n");
  divVecCPU(a,b,c);
  printf("Starting GPU operations...\n");

  double_c = test_vector<double,double>(a,b,divVecCUDA,double_time, repeats);
  printf("Time of double: %.16f\n",double_time/repeats);

  dsingle_c = test_vector<dsingle,dsingle>(a,b,divVecCUDA,dsingle_time, repeats);
  printf("Time of EP: %.16f\n",dsingle_time/repeats);

  float_c = test_vector<float,float>(a,b,divVecCUDA,float_time, repeats);
  printf("Time of float: %.16f\n",float_time/repeats);

  printf("ULP of double: %I64u\n",correctness(double_c, c));
  printf("ULP of EP: %I64u\n",correctness(dsingle_c, c));
  printf("ULP of float: %I64u\n",correctness(float_c, c));

  
  
  

  printf("Speed-up: %.16f\n", double_time/dsingle_time);

  //printf("%.16f\n",correctness(float_c, c)/correctness(dsingle_c, c));

  arr1 = correctnessArray(double_c, c);
  arr2 = correctnessArray(dsingle_c, c);
  arr3 = correctnessArray(float_c, c);
  if(saveData){
    f = fopen("div.csv", "w");
    fprintf(f, "a,b,c,double_c,epHi,epLo,epEval,float_c,doubleULP,epULP,floatULP\n");
    for( int i = 0; i < N; i++ ){
      fprintf(f, "%.16e,", a[i]);
      fprintf(f, "%.16e,", b[i]);
      fprintf(f, "%.16e,", c[i]);
      fprintf(f, "%.16e,", double_c[i]);
      fprintf(f, "%.16e,", dsingle_c[i].hi());
      fprintf(f, "%.16e,", dsingle_c[i].lo());
      fprintf(f, "%.16e,", dsingle_c[i].evaluate());
      fprintf(f, "%.16e,", float_c[i]);
      fprintf(f, "%I64u,", arr1[i]);
      fprintf(f, "%I64u,", arr2[i]);
      fprintf(f, "%I64u\n", arr3[i]);
    }
    fclose(f);
  }
  free(arr1);
  free(arr2);
  free(arr3);

  printf("---\npi\n");
  double double_pi = test_pi<double>(1000,double_time, repeats);
  printf("Time of double: %.16f\n",double_time/repeats);

  dsingle dsingle_pi = test_pi<dsingle>(1000,dsingle_time, repeats);
  printf("Time of EP: %.16f\n",dsingle_time/repeats);

  float float_pi = test_pi<float>(1000,float_time, repeats);
  printf("Time of float: %.16f\n",float_time/repeats);

  

  printf("ULP of double: %I64u\n",ULP(double_pi, double_pi));
  printf("ULP of EP: %I64u\n",ULP(dsingle_pi, double_pi));
  printf("ULP of float: %I64u\n",ULP(float_pi, double_pi));

  
  
  

  printf("Speed-up: %.16f\n", double_time/dsingle_time);


  free(dsingle_c);
  free(float_c);
  free(double_c);
  free(a);
  free(b);
  free(c);
  system("pause");
  return 0;
}