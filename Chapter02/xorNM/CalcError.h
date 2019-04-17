// The CalcError functor for XOR
  static const int nInput = 2;
  static const int nH1 = 1;
  static const int nOutput = 1;
  static const int nParam =
    (nOutput+nH1) // Neuron Offsets
    + (nInput*nH1) // connections from I to H1
    + (nH1*nOutput) // connections from H1 to O
    + (nInput*nOutput); // connections from I to O
  static const int exLen = nInput + nOutput;

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
      register Real h1 = p[index++];
      register Real o = p[index++];

      h1 += in[0] * p[index++];
      h1 += in[1] * p[index++];
      h1 = G(h1);

      o += in[0] * p[index++];
      o += in[1] * p[index++];
      o += h1 * p[index++];

      // calculate the square of the diffs
      o -= in[nInput];
      return o * o;
    }
  };
