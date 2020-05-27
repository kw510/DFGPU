struct dsingle{
  float x[2];
  //Constructors
  __host__ __device__ dsingle(float hi, float lo) { 
    x[0] = hi; x[1] = lo;
  }
  __host__ __device__ dsingle(float hi){
    x[0] = float(hi);
    x[1] = 0.0f;
  }
  __host__ __device__ dsingle(double hi){
    x[0] = float(hi);
    x[1] = float(hi - x[0]);
  }
  __host__ __device__ dsingle(int hi){
    x[0] = float(hi);
    x[1] = float(hi - x[0]);
  }
  __host__ __device__ dsingle(){
    x[0] = 0.0f;
    x[1] = 0.0f;
  }

  __host__ __device__ float hi(){
    return x[0];
  }

  __host__ __device__ float lo(){
    return x[1];
  }

  __host__ __device__ void print(){
    printf("x[0] = %.7e\n", x[0]);
    printf("x[1] = %.7e\n", x[1]);
  }
  __host__ __device__ void evaluate_print(){
    double out = this->evaluate();
    printf("%.13e\n", out);
  }
  __host__ __device__ double evaluate(){
    return (double) x[0] + (double) x[1];
  }

};