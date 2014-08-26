/* dense_mult.c
 * Copyright (C) 2013, Siddharth Gopal (gcdart AT gmail)
dgemm_ *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of version 2.1 of the GNU Lesser General Public License
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA, 02111-1307, USA,
 * or visit the GNU web site, www.gnu.org.
 */

#include <cblas.h>
#include <stdio.h>
#include <sys/time.h> 
#include <stdlib.h>
#include <math.h>

#define REP(i,n) for ( i = 0; i < (n); ++i ) 


void init_matrix(double* A, int dim1 , int dim2 ) {

  int mod = 100003, prod = 7 , e = 1 , i = 0, j = 0;

  REP(i,dim1) REP(j,dim2) {
    e = (e*prod + 1)%mod;
    A[i*dim2 + j] = e * .91739210437;
  }
}

void print_matrix (double *A , int dim1 , int dim2 ) {

  int i, j;

  REP(i,dim1) {
    REP(j,dim2) printf("%lf " , A[i*dim2 + j]);
    puts("");
  }
}


int main(int argc, char** argv){

  int m , n , k , i , j , u , nrep = 1 , cnt = 0;

  double *A , *B , *C , totaltime , error = 0;

  struct timeval before , after;

  if ( argc <= 1 || argc >= 6 ) {
    puts("./exe [nrep] <dim1> [dim2 dim3]");
    exit(0);
  }

  if ( argc == 2 ) {

    sscanf ( argv[1] , "%d" , &m );
    n = k = m;
  }
  else if ( argc == 3 ) {

    sscanf ( argv[1] , "%d" , &nrep );
    sscanf ( argv[2] , "%d" , &m );
    n = k = m;
  }
  else if ( argc == 4 ) {

    sscanf ( argv[1] , "%d" , &m );
    sscanf ( argv[2] , "%d" , &k );
    sscanf ( argv[3] , "%d" , &n );
  }
  else if ( argc == 5 ) {

    sscanf ( argv[1] , "%d" , &nrep );
    sscanf ( argv[2] , "%d" , &m );
    sscanf ( argv[3] , "%d" , &k );
    sscanf ( argv[4] , "%d" , &n );
  }

  A = (double *) malloc( sizeof(double) * m * k );
  B = (double *) malloc( sizeof(double) * k * n );
  C = (double *) malloc( sizeof(double) * m * n );

  init_matrix ( A , m , k );
  init_matrix ( B , k , n );

  printf ("Matrices Initialized [%dx%d , %dx%d , %dx%d ]\n" , m , k , k , n , m , n);

  printf ("Running %d Iteration(s) \n" , nrep );

  totaltime = 0;

  char NoTrans = 'N';

  double alpha = 1.0;

  double beta = 0.0;

  REP(i,nrep) {

    double runtime;

    gettimeofday ( &before , NULL );
  
    dgemm_(&NoTrans , &NoTrans , &m , &n , &k , &alpha , A , &m , B , &k , &beta , C , &m );

    gettimeofday ( &after , NULL );

    runtime = ( after.tv_usec - before.tv_usec )/1000000.0  + (after.tv_sec - before.tv_sec);

    if ( nrep == 1 || i > 0 ) 
      totaltime += runtime , cnt++;

    printf(" [Iteration %d Time : %lf ]\n" , i+1 , runtime );

    /*
    REP(i,m) REP(j,n) {
      double ans = 0;
      REP(u,k) ans += A[i*k+u]*B[u*n+j];
      error += (ans-C[i*n+j])*(ans-C[i*n+j]);
    }

    printf(" [Error = %lf ]\n" , error );
    */
  }

  printf("Average Time : %lf \n" , totaltime/cnt );

  return 0;
}
