#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
   int  numtasks, rank, ret; 

   ret = MPI_Init(&argc,&argv);
   if (ret != MPI_SUCCESS) {
     printf ("Error in MPI_Init()!\n");
     MPI_Abort(MPI_COMM_WORLD, ret);
     }

   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   printf ("Number of tasks= %d My rank= %d\n", numtasks,rank);

   /*******  do some work *******/

   MPI_Finalize();
}
