#include <stdio.h>
#include <math.h>
#include "headers/dsingle.h"
#include "error_functions.cu"

// Addition
// Algorithm 6 from Tight and rigourous error bounds. relative error < 3u²
__device__ dsingle operator+(dsingle a, dsingle b){
  float hi, lo,thi, tlo;
  // perform exact addition, with lo and tlo being the error term.
  two_sum(a.hi(), b.hi(),hi,lo);
  two_sum(a.lo(), b.lo(),thi,tlo);
  lo = lo + thi;
  quick_two_sum(hi,lo,hi,lo);
  lo = lo + tlo;
  quick_two_sum(hi,lo,hi,lo);

  return dsingle(hi,lo);
}
__device__ dsingle operator+(dsingle a, float b){
  float hi, lo;
  // perform exact addition
  two_sum(a.hi(),b,a.lo(),hi,lo);
  return dsingle(hi,lo);
}
__device__ dsingle operator+(float a, dsingle b){
  float hi, lo;
  // perform exact addition
  two_sum(a,b.hi(),b.lo(),hi,lo);
  return dsingle(hi,lo);
}

// Subtraction
__device__ dsingle operator-(dsingle a, dsingle b){
  float hi, lo;
  two_diff(a.hi(), b.hi(),hi,lo);

  float thi, tlo;
  two_diff(a.lo(), b.lo(),thi,tlo);

  lo = lo + thi;
  quick_two_sum(hi,lo,hi,lo);

  lo = lo + tlo;
  quick_two_sum(hi,lo,hi,lo);
  return dsingle(hi,lo);
}
__device__ dsingle operator-(dsingle a, float b){
  float hi, lo;
  two_sum(a.hi(),a.lo(),-b,hi,lo);
  return dsingle(hi,lo);
}
__device__ dsingle operator-(float a, dsingle b){
  float hi, lo;
  two_diff(a,b.hi(),b.lo(),hi,lo);
  return dsingle(hi,lo);
}
__device__ dsingle operator-(dsingle a){
  return dsingle(-a.hi(),-a.lo());
}

// Multiplication
__device__ dsingle operator*(dsingle a, dsingle b){
  float hi, lo;
  two_prod(a.hi(), b.hi(),hi,lo);

  //float t = a.lo() * b.lo();
  //t = fmaf(a.hi(),b.lo(),t);
  //t = fmaf(a.lo(),b.hi(),t);
  //lo = lo + t;

  lo = fmaf(a.lo(),b.lo(),lo);
  lo = fmaf(a.hi(),b.lo(),lo);
  lo = fmaf(a.lo(),b.hi(),lo);




  quick_two_sum(hi,lo,hi,lo);
  return dsingle(hi,lo);
}
__device__ dsingle operator*(dsingle a, float b){
  float hi, lo;
  two_prod(a.hi(), b,hi,lo);

  lo = fmaf(a.lo(), b, lo);

  quick_two_sum(hi,lo,hi,lo);
  return dsingle(hi,lo);
}
__device__ dsingle operator*(float a, dsingle b){
  return (b * a);
}


// Division
__device__ dsingle operator/(dsingle a, dsingle b){
  float hi, lo;
  hi = a.hi()/b.hi();

  float thi, tlo;
  two_prod(hi,b.hi(),thi,tlo);
  //lo = ((((a.hi() - thi) - tlo) + a.lo()) - hi*b.lo() ) / b.hi();
  //lo = fmaf(-hi,b.lo(),(((a.hi() - thi) - tlo) + a.lo())) / b.hi();
  lo = fmaf(-hi,b.lo(),(a.hi() - thi) + (a.lo() - tlo )) / b.hi();
  quick_two_sum(hi,lo,hi,lo);
  return dsingle(hi,lo);
}

__device__ dsingle operator/(dsingle a, float b){
  float hi, lo;
  hi = a.hi()/b;

  float thi, tlo;
  two_prod(hi,b,thi,tlo);
  lo = (((a.hi() - thi) - tlo) + a.lo())/b;

  quick_two_sum(hi,lo,hi,lo);
  return dsingle(hi,lo);
}
__device__ dsingle operator/(float a, dsingle b){
  float hi, lo;
  hi = a/b.hi();

  float thi, tlo;
  two_prod(hi,b.hi(),thi,tlo);
  //lo = ( ((a - thi) - tlo) - hi*b.lo() )/b.hi();
  lo = fmaf(-hi,b.lo(),((a - thi) - tlo))/b.hi();

  quick_two_sum(hi,lo,hi,lo);
  return dsingle(hi,lo);
}
/*

__global__ void addCUDA(dsingle *a, dsingle *b, dsingle *c){
  *c = *a / *b;
}
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

  double truev = x / y;
  printf("%.16f : native double\n",truev);
  printf("%.16f : extended\n", c.evaluate());
  printf("%.16f : diffrence\n", c.evaluate() - truev);
  printf("---\n");

  cudaFree( da );
  cudaFree( db );
  cudaFree( dc );
}
int main(int argc, char const *argv[])
{
  test_addition((1.0/3),(2.0f));
  return 0;
}*/