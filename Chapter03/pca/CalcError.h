  static const int nInput = 2;
  static const int nH1 = 1;
  static const int nOutput = nInput;
  static const int nParam = 
    (nOutput+nH1) // Neuron Offsets
    + (nInput*nH1) // connections from I to H1
    + (nH1*nOutput); // connections from H1 to O
  static const int exLen = nInput;
  
  struct CalcError {
    const Real* examples;
    const Real* p;
    const int nInput;
    const int exLen;
    
  CalcError( const Real* _examples, const  Real* _p,
             const int _nInput, const int _exLen)
  : examples(_examples), p(_p), nInput(_nInput), exLen(_exLen) {};
    
    __device__ __host__
    Real operator()(unsigned int tid)
    {
      const register Real* in = &examples[tid * exLen];
      register int index=0;

      register Real h0 = p[index++];
      
      for(int i=0; i < nInput; i++) {
        register Real input=in[i];
        h0 += input * p[index++];
      }

      register Real sum = 0.;
      for(int i=0; i < nInput; i++) {
        register Real o = p[index++];
        o += h0 * p[index++];
        o -= in[i];
        sum += o*o;
      }
      return sum;
    }
  };


