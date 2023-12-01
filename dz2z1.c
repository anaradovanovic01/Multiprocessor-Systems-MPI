#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define MASTER 0

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

int prime_number(int n)
{
  int i;
  int j;
  int prime;
  int total, total_master;
  int rank, size;
  int start, end, chunk;

  total = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int start_arr[size], end_arr[size];
  if(rank == MASTER) {
    chunk = (n + size - 1) / size;
    for(int i = 0; i < size; i++) {
      start_arr[i] = 2 + i * chunk;
      end_arr[i] = start_arr[i] + chunk < n ? start_arr[i] + chunk : n;
    }
  }
  MPI_Scatter(start_arr, 1, MPI_INT, &start, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Scatter(end_arr, 1, MPI_INT, &end, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

  if(n == 2 && rank == MASTER) end = 3;
  
  for (i = start; i < end; i++)
  {
    prime = 1;
    for (j = 2; j < i; j++)
    {
      if ((i % j) == 0)
      {
        prime = 0;
        break;
      }
    }
    total = total + prime;
  }
  MPI_Reduce(&total, &total_master, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  return total_master;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

void test(int n_lo, int n_hi, int n_factor);

int main(int argc, char *argv[])
{
  int n_factor;
  int n_hi;
  int n_lo;
  double starttime, endtime;

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(rank == MASTER) {
    timestamp();
    printf("\n");
    printf("PRIME TEST\n");
    starttime = MPI_Wtime();
  }

  if (argc != 4)
  {
    n_lo = 1;
    n_hi = 131072;
    n_factor = 2;
  }
  else
  {
    n_lo = atoi(argv[1]);
    n_hi = atoi(argv[2]);
    n_factor = atoi(argv[3]);
  }
  
  test(n_lo, n_hi, n_factor);
  
  if(rank == MASTER) {
    printf("\n");
    printf("PRIME_TEST\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    timestamp();
    endtime = MPI_Wtime();
    printf("Time elapsed: %lf\n", endtime - starttime);
  }

  MPI_Finalize();
  
  return 0;
}

void test(int n_lo, int n_hi, int n_factor)
{
  int i;
  int n;
  int primes;
  double ctime;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
  if(rank == MASTER) {
    printf("\n");
    printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
    printf("\n");
    printf("         N        Pi          Time\n");
    printf("\n");
  }

  n = n_lo;

  while (n <= n_hi)
  {
    if(rank == MASTER) ctime = cpu_time();

    primes = prime_number(n);

    if(rank == MASTER) {
      ctime = cpu_time() - ctime;
      printf("  %8d  %8d  %14f\n", n, primes, ctime);
    }

    n = n * n_factor;
  
  }

  return;
}
