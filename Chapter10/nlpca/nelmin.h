#ifndef NELMIN_H
#define NELMIN_H

// Nelder-Mead Minimization Algorithm ASA047
// from the Applied Statistics Algorithms available
// in STATLIB. Adapted from the C version by J. Burkhardt
// http://people.sc.fsu.edu/~jburkardt/c_src/asa047/asa047.html

template <typename Real>
void nelmin ( Real (*fn)(Real*), int n, Real start[], Real xmin[], 
	      Real *ynewlo, Real reqmin, Real step[], int konvge, int kcount, 
	      int *icount, int *numres, int *ifault )
{
  const Real ccoeff = 0.5;
  const Real ecoeff = 2.0;
  const Real eps = 0.001;
  const Real rcoeff = 1.0;
  int ihi,ilo,jcount,l,nn;
  Real del,dn,dnn;
  Real rq,x,y2star,ylo,ystar,z;

  //  Check the input parameters.
  if ( reqmin <= 0.0 ) { *ifault = 1; return; }
  if ( n < 1 ) { *ifault = 1; return; }
  if ( konvge < 1 ) { *ifault = 1; return; }

  vector<Real> p(n*(n+1));
  vector<Real> pstar(n);
  vector<Real> p2star(n);
  vector<Real> pbar(n);
  vector<Real> y(n+1);

  *icount = 0;
  *numres = 0;

  jcount = konvge; 
  dn = ( Real ) ( n );
  nn = n + 1;
  dnn = ( Real ) ( nn );
  del = 1.0;
  rq = reqmin * dn;
  //  Initial or restarted loop.
  for ( ; ; ) {
    for (int i = 0; i < n; i++ ) { p[i+n*n] = start[i]; }
    y[n] = (*fn)( start );
    *icount = *icount + 1;
    
    for (int j = 0; j < n; j++ ) {
      x = start[j];
      start[j] = start[j] + step[j] * del;
      for (int i = 0; i < n; i++ ) { p[i+j*n] = start[i]; }
      y[j] = (*fn)( start );
      *icount = *icount + 1;
      start[j] = x;
    }
    //  The simplex construction is complete.
    //                    
    //  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
    //  the vertex of the simplex to be replaced.
    ylo = y[0];
    ilo = 0;
    
    for (int i = 1; i < nn; i++ ) {
      if ( y[i] < ylo ) { ylo = y[i]; ilo = i; }
    }
    //  Inner loop.
    for ( ; ; ) {
      if ( kcount <= *icount ) { break; }
      *ynewlo = y[0];
      ihi = 0;
      
      for (int i = 1; i < nn; i++ ) {
        if ( *ynewlo < y[i] ) { *ynewlo = y[i]; ihi = i; }
      }
      //  Calculate PBAR, the centroid of the simplex vertices
      //  excepting the vertex with Y value YNEWLO.
      for (int i = 0; i < n; i++ ) {
        z = 0.0;
        for (int j = 0; j < nn; j++ ) { z = z + p[i+j*n]; }
        z = z - p[i+ihi*n];  
        pbar[i] = z / dn;
      }
      //  Reflection through the centroid.
      for (int i = 0; i < n; i++ ) {
        pstar[i] = pbar[i] + rcoeff * ( pbar[i] - p[i+ihi*n] );
      }
      ystar = (*fn)( &pstar[0] );
      *icount = *icount + 1;
      //  Successful reflection, so extension.
      if ( ystar < ylo ) {
        for (int i = 0; i < n; i++ ) {
          p2star[i] = pbar[i] + ecoeff * ( pstar[i] - pbar[i] );
        }
        y2star = (*fn)( &p2star[0] );
        *icount = *icount + 1;
	//  Check extension.
        if ( ystar < y2star ) {
          for (int i = 0; i < n; i++ ) { p[i+ihi*n] = pstar[i]; }
          y[ihi] = ystar;
        } else { //  Retain extension or contraction.
          for (int i = 0; i < n; i++ ) { p[i+ihi*n] = p2star[i]; }
          y[ihi] = y2star;
        }
      } else { //  No extension.
        l = 0;
        for (int i = 0; i < nn; i++ ) {
	  if ( ystar < y[i] ) l += 1;
        }
	
        if ( 1 < l ) {
          for (int i = 0; i < n; i++ ) { p[i+ihi*n] = pstar[i]; }
          y[ihi] = ystar;
        }
	//  Contraction on the Y(IHI) side of the centroid.
        else if ( l == 0 ) {
          for (int i = 0; i < n; i++ ) {
            p2star[i] = pbar[i] + ccoeff * ( p[i+ihi*n] - pbar[i] );
          }
          y2star = (*fn)( &p2star[0] );
          *icount = *icount + 1;
	  //  Contract the whole simplex.
          if ( y[ihi] < y2star ) {
            for (int j = 0; j < nn; j++ ) {
              for (int i = 0; i < n; i++ ) {
                p[i+j*n] = ( p[i+j*n] + p[i+ilo*n] ) * 0.5;
                xmin[i] = p[i+j*n];
              }
              y[j] = (*fn)( xmin );
              *icount = *icount + 1;
            }
            ylo = y[0];
            ilo = 0;
	    
            for (int i = 1; i < nn; i++ ) {
              if ( y[i] < ylo ) { ylo = y[i]; ilo = i; }
            }
            continue;
          }
	  //
	  //  Retain contraction.
	  //
          else {
            for (int i = 0; i < n; i++ ) {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
        }
	//
	//  Contraction on the reflection side of the centroid.
	//
        else if ( l == 1 ) {
          for (int i = 0; i < n; i++ ) {
            p2star[i] = pbar[i] + ccoeff * ( pstar[i] - pbar[i] );
          }
          y2star = (*fn)( &p2star[0] );
          *icount = *icount + 1;
	  //
	  //  Retain reflection?
	  //
          if ( y2star <= ystar ) {
            for (int i = 0; i < n; i++ ) { p[i+ihi*n] = p2star[i]; }
            y[ihi] = y2star;
          }
          else {
            for (int i = 0; i < n; i++ ) { p[i+ihi*n] = pstar[i]; }
            y[ihi] = ystar;
          }
        }
      }
      //
      //  Check if YLO improved.
      //
      if ( y[ihi] < ylo ) { ylo = y[ihi]; ilo = ihi; }
      jcount = jcount - 1;
      
      if ( 0 < jcount ) { continue; }
      //
      //  Check to see if minimum reached.
      //
      if ( *icount <= kcount ) {
        jcount = konvge;
	
        z = 0.0;
        for (int i = 0; i < nn; i++ ) { z = z + y[i]; }
        x = z / dnn;
	
        z = 0.0;
        for (int i = 0; i < nn; i++ ) {
          z = z + pow ( y[i] - x, 2 );
        }
	
        if ( z <= rq ) {
          break;
        }
      }
    }
    //
    //  Factorial tests to check that YNEWLO is a local minimum.
    //
    for (int i = 0; i < n; i++ ) { xmin[i] = p[i+ilo*n]; }
    *ynewlo = y[ilo];
    
    if ( kcount < *icount ) { *ifault = 2; break; }

    *ifault = 0;

    for (int i = 0; i < n; i++ ) {
      del = step[i] * eps;
      xmin[i] = xmin[i] + del;
      z = (*fn)( xmin );
      *icount = *icount + 1;
      if ( z < *ynewlo ) { *ifault = 2; break; }
      xmin[i] = xmin[i] - del - del;
      z = (*fn)( xmin );
      *icount = *icount + 1;
      if ( z < *ynewlo ) { *ifault = 2; break; }
      xmin[i] = xmin[i] + del;
    }
    
    if ( *ifault == 0 ) { break; }
    //
    //  Restart the procedure.
    //
    for (int i = 0; i < n; i++ ) { start[i] = xmin[i]; }
    del = eps;
    *numres = *numres + 1;
  }
  return;
}
#endif
