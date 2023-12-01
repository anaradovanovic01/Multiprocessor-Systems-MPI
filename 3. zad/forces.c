
/*
 *  Compute forces and accumulate the virial and the potential
 */
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MASTER 0

extern double epot, vir;

void forces(int npart, double x[], double f[], double side, double rcoff)
{
  int i, j;
  double sideh, rcoffs;
  double xi, yi, zi, fxi, fyi, fzi, xx, yy, zz;
  double rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148;
  double forcex, forcey, forcez;

  int rank, size;
  double vir2 = 0, epot2 = 0, *f2;
  double vir_master, epot_master;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(rank == MASTER) f2 = malloc(__SIZEOF_DOUBLE__* npart * 3);

  vir = 0.0;
  epot = 0.0;
  sideh = 0.5 * side;
  rcoffs = rcoff * rcoff;

  for (i = rank * 3; i < npart * 3; i += 3 * size)
  {
    xi = x[i];
    yi = x[i + 1];
    zi = x[i + 2];
    fxi = 0.0;
    fyi = 0.0;
    fzi = 0.0;

    for (j = i + 3; j < npart * 3; j += 3)
    {
      xx = xi - x[j];
      yy = yi - x[j + 1];
      zz = zi - x[j + 2];
      if (xx < -sideh)
        xx += side;
      if (xx > sideh)
        xx -= side;
      if (yy < -sideh)
        yy += side;
      if (yy > sideh)
        yy -= side;
      if (zz < -sideh)
        zz += side;
      if (zz > sideh)
        zz -= side;
      rd = xx * xx + yy * yy + zz * zz;

      if (rd <= rcoffs)
      {
        rrd = 1.0 / rd;
        rrd2 = rrd * rrd;
        rrd3 = rrd2 * rrd;
        rrd4 = rrd2 * rrd2;
        rrd6 = rrd2 * rrd4;
        rrd7 = rrd6 * rrd;
        epot2 += (rrd6 - rrd3);
        r148 = rrd7 - 0.5 * rrd4;
        vir2 -= rd * r148;
        forcex = xx * r148;
        fxi += forcex;
        f[j] -= forcex;
        forcey = yy * r148;
        fyi += forcey;
        f[j + 1] -= forcey;
        forcez = zz * r148;
        fzi += forcez;
        f[j + 2] -= forcez;
      }
    }
    f[i] += fxi;
    f[i + 1] += fyi;
    f[i + 2] += fzi;
  }

  MPI_Reduce(f, f2, npart * 3, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
  if(rank == MASTER) {
    memcpy(f, f2, npart*3*__SIZEOF_DOUBLE__);
    free(f2);  
  }
  MPI_Bcast(f, npart*3, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
  
  MPI_Reduce(&vir2, &vir_master, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
  vir += vir_master;
  MPI_Bcast(&vir, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

  MPI_Reduce(&epot2, &epot_master, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
  epot += epot_master;
  MPI_Bcast(&epot, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

}
