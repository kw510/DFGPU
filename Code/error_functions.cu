/// basic functions
// two sum
__device__ void two_sum(float a, float b, float &hi, float &lo){
  hi = a + b;  // best guess
  float v = hi - a;
  lo = (a - (hi - v)) + (b - v);
}

__device__ void two_sum(float a, float b, float c, float &hi, float &lo){
  float s,t,u;
  two_sum(b,c,s,t);
  two_sum(a,s,hi,u);
  lo = u + t;
  two_sum(hi,lo,hi,lo);
}

__device__ void two_sum(float a, float b, float c, float d, float &hi, float &lo){
  float t0,t1,t2,t3;
  two_sum(a,b,t0,t1);
  two_sum(t0,c,t0,t2);
  two_sum(t0,d,hi,t3);
  t0 = t1 + t2;
  lo = t0 + t3;
}
/*
* Unchecked requirement
* |a| > |b|
*/
__device__ void quick_two_sum(float a, float b, float &hi, float &lo){
  hi = a + b;  // floating point guess
  lo = b - (hi - a); // error calculation
}
/*
* Unchecked requirement
* |a| > |b| > |c|
*/
__device__ void quick_two_sum(float a, float b,float c, float &hi, float &lo){
  float s,t,u;
  quick_two_sum(b,c,s,t);
  quick_two_sum(a,s,hi,u);
  lo = u + t;;
  quick_two_sum(hi,lo,hi,lo);
}
/*
* Unchecked requirement
* |a| > |b| > |c| > |d|
*/
__device__ void quick_two_sum(float a, float b, float c, float d, float &hi, float &lo){
  float t0,t1,t2,t3;
  quick_two_sum(a,b,t0,t1);
  quick_two_sum(t0,c,t0,t2);
  quick_two_sum(t0,d,hi,t3);
  t0 = t1 + t2;
  lo = t0 + t3;
}


__device__ void quick_two_diff(float a, float b, float &hi, float &lo) {
  hi = a - b;
  lo = (a - hi) - b;
}



__device__ void two_diff(float a, float b, float &hi, float &lo) {
  hi = a - b;
  float v = hi - a;
  lo = (a - (hi - v)) - (b + v);

  /*hi = a - b; 
  float a1 = hi + b;
  float b1 = hi - a1;
  lo = (a - a1) - (b + b1);*/
}
__device__ void two_diff(float a, float b, float c, float &hi, float &lo){
  float s,t,u;
  two_diff(-b,c,s,t);
  two_sum(a,s,hi,u);
  lo = u + t;;
  two_sum(hi,lo,hi,lo);
}

__device__ void two_prod(float a, float b, float &hi, float &lo){
  hi = a * b;
  lo = fmaf(a, b, -hi);
}