export OPENBLAS_LIBDIR="/opt/OpenBLAS/lib"
export OPENBLAS_INCDIR="/opt/OpenBLAS/include"

gcc -static \
-O4 -msse2 -msse3 -msse4 \
-I $OPENBLAS_INCDIR \
dense_mult.c \
-L $OPENBLAS_LIBDIR -lopenblas -fopenmp \
-o ap.out


cp ap.out a.out
