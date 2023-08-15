#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************

__global__ void AssignRequests(int *CentreRequests,int *RequestFrequency,int *facility,int *req_start,int *req_slots,int *capacity,int *total, int *success,int *tot,int *suc,int R,int N)
{
  int id=blockDim.x*blockIdx.x+threadIdx.x; 
  if(id<N*max_P)
  {
    int id1=id/max_P,id2=id%max_P; // id1 resembles the centre id whereas id2 resembles the facility within that centre
    if(id2<facility[id1])  // Facility id should be less than the number of facilities in that centre
    {
      int cap=capacity[id1*max_P+id2],count=0,start,end,i,j; // Capacity the facility
      int assign[24]; // This array is used to assign slots to requests. It's indexe's value tells how many requests are running on that slot.
      for(i=0;i<24;i++)
      {
        assign[i]=0;
      }
      for(i=0;i<RequestFrequency[id1*max_P+id2];i++) // A loop that iterates over all the requests on a particular facility.
      {
        start=req_start[CentreRequests[i+id1*max_P*R+id2*R]]-1; // Starting slot needed by a request
        end=start+req_slots[CentreRequests[i+id1*max_P*R+id2*R]];  // End slot needed by a request
        for(j=start;j<end;j++)
        {
          if(assign[j]>=cap) // It means if any slot is at it's full capacity
          {
            break;
          }
        }
        if(j==end) // It means no slot in the range start to end is on it's full capacity. Hence, the request can be fullfilled.
        {
          for(j=start;j<end;j++)
          {
            assign[j]++; // Increasing all slots by 1 cause they are running a new request.
          }
          count++; // Stores the number of successful requests.
        }
      }
      atomicAdd(&total[id1],RequestFrequency[id1*max_P+id2]); // Stores the total number of requests per center
      atomicAdd(&success[id1],count); // Stores successful requests per center
      atomicAdd(tot,RequestFrequency[id1*max_P+id2]); // Stores the total number of requests overall
      atomicAdd(suc,count); // Stores successful requests overall
    }
  }
}
//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[i*max_P+j] );
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[i*max_P+j]);   
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		

    int *d_req_id,*d_facility,*d_total,*d_success,*d_capacity,*d_t,*d_s,*d_req_cen, *d_req_fac, *d_req_start, *d_req_slots,*CentreRequests,*RequestFrequency,*d_CentreRequests,*d_RequestFrequency;
    RequestFrequency=(int*)malloc((N *max_P)* sizeof (int));
    memset(RequestFrequency, 0, (N*max_P)*sizeof(int));

    cudaMalloc(&d_t, sizeof(int));  // Variable to store total number of requests
    cudaMalloc(&d_s, sizeof(int));  // Variable to store total successful requests
    cudaMalloc(&d_total, (N)*sizeof(int));  // Variable to store total requests per centre
    cudaMalloc(&d_success, (N)*sizeof(int));  // Variable to store successful requests per centre
    cudaMalloc(&d_facility, (N)*sizeof(int));
    cudaMalloc(&d_req_id, (R)*sizeof(int));
    cudaMalloc(&d_req_cen, (R)*sizeof(int));
    cudaMalloc(&d_req_fac, (R)*sizeof(int));
    cudaMalloc(&d_req_start, (R)*sizeof(int));
    cudaMalloc(&d_req_slots, (R)*sizeof(int));
    cudaMalloc(&d_RequestFrequency, (N*max_P)*sizeof(int));  // Variable to store the number of requests per facility
    cudaMalloc(&d_capacity, (max_P*N)*sizeof(int));

    cudaMemcpy(d_facility, facility, (N)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_id, req_id, (R)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen, req_cen, (R)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac, req_fac, (R)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start, req_start, (R)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots,req_slots, (R)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity,capacity, (N*max_P)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(&d_total, 0, N*sizeof(int));
    cudaMemset(&d_success, 0, N*sizeof(int));
    int ma=0; // Variable to store maximum number of requests at a particluar facility for given input
    for(int i=0;i<R;i++)
    {
      RequestFrequency[req_cen[i]*max_P+req_fac[i]]++;  // Incrementing the frequency of requests in a particular facility 
      if(ma<RequestFrequency[req_cen[i]*max_P+req_fac[i]])
      {
        ma=RequestFrequency[req_cen[i]*max_P+req_fac[i]];  // Updating maximum
      }
    }
    memset(RequestFrequency, 0, (N*max_P)*sizeof(int)); // Initializing it to all 0's cause we need to use it again.
    CentreRequests=(int*)malloc((N *max_P * ma)* sizeof (int)); // A variable used to store request id's of requests per facility.
    cudaMalloc(&d_CentreRequests, (ma * N * max_P)*sizeof(int));
    for(int i=0;i<R;i++)
    {
      // Here since we are iterating serially and requests id's are 0 to R-1, they would already be sorted. Hence, no sorting needed.
      CentreRequests[req_cen[i]*ma*max_P+req_fac[i]*ma+RequestFrequency[req_cen[i]*max_P+req_fac[i]]]=req_id[i];// Storing requests id's corresponding to the facility they are incident on.
      RequestFrequency[req_cen[i]*max_P+req_fac[i]]++;// It now resemebles the next index in CentreRequests where id's need to be added.
    }
    cudaMemcpy(d_CentreRequests,CentreRequests, N*max_P*ma*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RequestFrequency,RequestFrequency, (N*max_P)*sizeof(int), cudaMemcpyHostToDevice);
    // This kernel basically runs all facility rooms in parallel and assign the requests that they can execute.
    AssignRequests<<<ceil((float)(N*max_P)/1024.0),1024>>>(d_CentreRequests,d_RequestFrequency,d_facility,d_req_start,d_req_slots,d_capacity,d_total,d_success,d_t,d_s,ma,N);
    cudaMemcpy(tot_reqs,d_total,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(succ_reqs,d_success,N*sizeof(int),cudaMemcpyDeviceToHost);
    int *t,*s;
    t=(int*)malloc(sizeof (int)); // Variable to stores total requests in CPU
    s=(int*)malloc(sizeof (int));  // Variable to stores successful requests in CPU
    cudaMemcpy(s,d_s,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(t,d_t,sizeof(int),cudaMemcpyDeviceToHost);
    success=*s;
    fail=*t-success;// Failed requets = Total requests - Successful requests
    //********************************

    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}