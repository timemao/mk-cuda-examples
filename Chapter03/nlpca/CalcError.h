  static const int nInput = 2;
  static const int nH1 = 4;
  static const int nH2 = 1;
  static const int nH3 = 4;
  static const int nOutput = nInput;
  static const int nParam = 
    (nOutput+nH1+nH2+nH3) // Neuron Offsets
    + (nInput*nH1) // connections from I to H1
    + (nH1*nH2) // connections from H1 to H2
    + (nH2*nH3) // connections from H2 to H3
    + (nH3*nOutput); // connections from H3 to O
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
      
      register Real h2_0 = p[index++]; // bottleneck neuron
      {
	register Real h1_0 = p[index++];
	register Real h1_1 = p[index++];
	register Real h1_2 = p[index++];
	register Real h1_3 = p[index++];
	for(int i=0; i < nInput; i++) {
	  register Real input=in[i];
	  h1_0 += input * p[index++]; h1_1 += input * p[index++];
	  h1_2 += input * p[index++]; h1_3 += input * p[index++];
	}
	h1_0 = G(h1_0); h1_1 = G(h1_1);
	h1_2 = G(h1_2); h1_3 = G(h1_3);
	
	h2_0 += p[index++] * h1_0; h2_0 += p[index++] * h1_1;
	h2_0 += p[index++] * h1_2; h2_0 += p[index++] * h1_3;
      }
      register Real h3_0 = p[index++];
      register Real h3_1 = p[index++];
      register Real h3_2 = p[index++];
      register Real h3_3 = p[index++];
      h3_0 += p[index++] * h2_0; h3_1 += p[index++] * h2_0;
      h3_2 += p[index++] * h2_0; h3_3 += p[index++] * h2_0;
      h3_0 = G(h3_0); h3_1 = G(h3_1);
      h3_2 = G(h3_2); h3_3 = G(h3_3);
      
      register Real sum = 0.;
      for(int i=0; i < nOutput; i++) {
	register Real o = p[index++];
	o += h3_0 * p[index++]; o += h3_1 * p[index++];
	o += h3_2 * p[index++]; o += h3_3 * p[index++];
	o -= in[i];
	sum += o*o;
      }
      return sum;
    }
  };
