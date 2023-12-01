/*
 *  Compute forces and accumulate the virial and the potential
 */
#include <stdio.h>
#include <mpi.h>

#define MASTER 0
#define END_TAG 1000
#define ITER_TAG 1001
#define VIR_TAG 1002
#define EPOT_TAG 1003
#define I_TAG 1004
#define F_TAG 1005

extern double epot, vir;

void forces(int npart, double x[], double f[], double side, double rcoff)
{
  int i, j;
  double sideh, rcoffs;
  double xi, yi, zi, fxi, fyi, fzi, xx, yy, zz;
  double rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148;
  double forcex, forcey, forcez;
  int size, rank, cnt, cnt_recv;
  double vir2, epot2, f2[npart*3];
  MPI_Status status;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  vir = 0.0;
  epot = 0.0;
  sideh = 0.5 * side;
  rcoffs = rcoff * rcoff;

  if (rank == MASTER) {
    MPI_Request request;
    cnt = 0;
    cnt_recv = 0;
    for(int k = 1; k < size; k++, cnt += 3) {
      MPI_Send(&cnt, 1, MPI_INT, k, ITER_TAG, MPI_COMM_WORLD);
    }
    
    while(cnt_recv < npart * 3) {
      MPI_Recv(&vir2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, VIR_TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&epot2, 1, MPI_DOUBLE, status.MPI_SOURCE, EPOT_TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&i, 1, MPI_INT, status.MPI_SOURCE, I_TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(f2+i, npart*3 - i, MPI_DOUBLE, status.MPI_SOURCE, F_TAG, MPI_COMM_WORLD, &status);

      if(cnt < npart*3) {
         MPI_Send(&cnt, 1, MPI_INT, status.MPI_SOURCE, ITER_TAG, MPI_COMM_WORLD);
          cnt += 3;
      }

      vir += vir2;
      epot += epot2;
      for(int b=i; b < npart * 3; b++){
        f[b] += f2[b];
      }
      cnt_recv += 3;
    }
    for(int k = 1; k < size && k < npart*3; k++, cnt++)
      MPI_Send(&cnt, 1, MPI_INT, k, END_TAG, MPI_COMM_WORLD);
  }
  else {
    while(1) {
      MPI_Recv(&i, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if(status.MPI_TAG == END_TAG) break;

      xi = x[i];
      yi = x[i + 1];
      zi = x[i + 2];
      fxi = 0.0;
      fyi = 0.0;
      fzi = 0.0;
      vir2 = 0.0;
      epot2 = 0.0;
      for(int i= 0; i < npart*3; i++) f2[i] = 0;

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
          f2[j] -= forcex;
          forcey = yy * r148;
          fyi += forcey;
          f2[j + 1] -= forcey;
          forcez = zz * r148;
          fzi += forcez;
          f2[j + 2] -= forcez;
        }
      }
      f2[i] += fxi;
      f2[i + 1] += fyi;
      f2[i + 2] += fzi;
      MPI_Send(&vir2, 1, MPI_DOUBLE, MASTER, VIR_TAG, MPI_COMM_WORLD);
      MPI_Send(&epot2, 1, MPI_DOUBLE, MASTER, EPOT_TAG, MPI_COMM_WORLD);
      MPI_Send(&i, 1, MPI_INT, MASTER, I_TAG, MPI_COMM_WORLD);
      MPI_Send(f2+i, npart*3 - i, MPI_DOUBLE, MASTER, F_TAG, MPI_COMM_WORLD);
    }
  }
  MPI_Bcast(f, npart*3, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
}