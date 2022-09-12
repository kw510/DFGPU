struct tsingle{
  float x[3];
  //Constructors
  __host__ __device__ tsingle(){
    x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
  }
  __host__ __device__ tsingle(float hi, float mid, float lo) { 
    x[0] = hi; x[1] = mid; x[2] = lo; 
  }
  __host__ __device__ tsingle(float hi, float mid){
    x[0] = hi; x[1] = mid; x[2] = 0.0;
  }
  __host__ __device__ tsingle(float hi) {
    x[0] = hi; x[1] = 0.0; x[2] = 0.0; 
  }
  __host__ __device__ tsingle(double hi){
    x[0] = hi;
    x[1] = hi - x[0];
    x[2] = (hi - x[0]) - x[1];
  }

  __host__ __device__ float hi(){
    return x[0];
  }
  __host__ __device__ float mid(){
    return x[1];
  }
  __host__ __device__ float lo(){
    return x[2];
  }

  __host__ __device__ void print(){
    printf("x[0] = %.7e\n", x[0]);
    printf("x[1] = %.7e\n", x[1]);
    printf("x[2] = %.7e\n", x[2]);
  }
  __host__ __device__ void evaluate_print(){
    double out = this->evaluate();
    printf("%.15e\n", out);
  }
  __host__ __device__ double evaluate(){
    return (double) x[0] + (double) x[1] + (double) x[2];
  }
  __host__ __device__ double hi_mid_evaluate(){
    return (double) x[0] + (double) x[1];
  }

};