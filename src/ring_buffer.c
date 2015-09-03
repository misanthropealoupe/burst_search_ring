#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "ring_buffer.h"

#ifndef max
  #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
  #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define DM0 4148.8
#define NOISE_PERIOD 64
#define SIG_THRESH 30.0
#define THREAD 8
#define OMP_THREADS 8

// float **matrix(long n, long m)
// {

//   float *vec=(float *)malloc(n*m*sizeof(float));
//   float **mat=(float **)malloc(n*sizeof(float *));
//   mat[0] = vec;
//   for (long i=1;i<n;i++) 
//     mat[i]=mat[i -1] + m;
//   return mat;
// }

float **matrix(long n, long m)
{

  float *vec=(float *)malloc(n*m*sizeof(float));
  float **mat=(float **)malloc(n*sizeof(float *));
  for (long i=0;i<n;i++) 
    mat[i]=vec+i*m;
  return mat;
}

float *vector(int n)
{
  float *vec=(float *)malloc(sizeof(float)*n);
  assert(vec);
  memset(vec,0,n*sizeof(float));
  return vec;
}

void get_omp_iminmax(int n,  int *imin, int *imax)
{
  int nthreads=omp_get_num_threads();
  int myid=omp_get_thread_num();
  int bs=n/nthreads;
  *imin=myid*bs;
  *imax=(myid+1)*bs;
  if (*imax>n)
    *imax=n;

}

void clean_rows_2pass(float *vec, size_t nchan, size_t ndata)
{
  float **dat=(float **)malloc(sizeof(float *)*nchan);
  for (int i=0;i<nchan;i++)
    dat[i]=vec+i*ndata;
  
  float *tot=vector(ndata);
  memset(tot,0,sizeof(tot[0])*ndata);

  //find the common mode based on averaging over channels.
  //inner loop is the natural one to parallelize over, but for some
  //architectures/compilers doing so with a omp for is slow, hence
  //rolling my own.
#pragma omp parallel shared(ndata,nchan,dat,tot) default(none)
  {
    int imin,imax;
    get_omp_iminmax(ndata,&imin,&imax);
    
    for (int i=0;i<nchan;i++)
      for (int j=imin;j<imax;j++)
  tot[j]+=dat[i][j];
    
    for (int j=imin;j<imax;j++)
      tot[j]/=nchan;
  }
  //#define BURST_DUMP_DEBUG
#ifdef BURST_DUMP_DEBUG
  FILE *outfile=fopen("common_mode_1.dat","w");
  fwrite(tot,sizeof(float),ndata,outfile);
  fclose(outfile);
#endif

  float *amps=vector(nchan);
  memset(amps,0,sizeof(amps[0])*nchan);
  float totsqr=0;
  for (int i=0;i<ndata;i++)
    totsqr+=tot[i]*tot[i];

  //find the best-fit amplitude for each channel
#pragma omp parallel for shared(ndata,nchan,dat,tot,amps,totsqr) default(none)
  for (int i=0;i<nchan;i++) {
    float myamp=0;
    for (int j=0;j<ndata;j++)
      myamp+=dat[i][j]*tot[j];
    myamp/=totsqr;
    //for (int j=0;j<ndata;j++)
    //   dat[i][j]-=tot[j]*myamp;
    amps[i]=myamp;    
  }

#ifdef BURST_DUMP_DEBUG
  outfile=fopen("mean_responses1.txt","w");
  for (int i=0;i<nchan;i++)
    fprintf(outfile,"%12.4f\n",amps[i]);
  fclose(outfile);
#endif

  //decide that channels with amplitude between 0.5 and 1.5 are the good ones.
  //recalculate the common mode based on those guys, with appropriate calibration
  memset(tot,0,sizeof(tot[0])*ndata);
  float amp_min=0.5;
  float amp_max=1.5;
#pragma omp parallel shared(ndata,nchan,amps,dat,tot,amp_min,amp_max) default(none)
  {
    int imin,imax;
    get_omp_iminmax(ndata,&imin,&imax);

    for (int i=0;i<nchan;i++) {
      if ((amps[i]>amp_min)&&(amps[i]<amp_max))
  for (int j=imin;j<imax;j++)
    tot[j]+=dat[i][j]/amps[i];
    }
  }
  float tot_sum=0;
  for (int i=0;i<nchan;i++)
    if ((amps[i]>amp_min)&&(amps[i]<amp_max))
      tot_sum+=1./amps[i];
  totsqr=0;
  for (int i=0;i<ndata;i++) {
    tot[i]/=tot_sum;
    totsqr+=tot[i]*tot[i];
  }
  
#ifdef BURST_DUMP_DEBUG
  outfile=fopen("common_mode_2.dat","w");
  fwrite(tot,sizeof(float),ndata,outfile);
  fclose(outfile);

  {
    float *chansum=vector(nchan);
    float *chansumsqr=vector(nchan);
#pragma omp parallel for
    for (int i=0;i<nchan;i++)
      for (int j=0;j<ndata;j++) {
  chansum[i]+=dat[i][j];
  chansumsqr[i]+=dat[i][j]*dat[i][j];
      }
    outfile=fopen("chan_variances_pre.txt","w");
    for (int i=0;i<nchan;i++)
      fprintf(outfile,"%12.6e\n",sqrt(chansumsqr[i]-chansum[i]*chansum[i]/ndata));
    fclose(outfile);
    free(chansum);
    free(chansumsqr);
  }
#endif



  memset(amps,0,sizeof(amps[0])*nchan);
#pragma omp parallel for shared(ndata,nchan,amps,dat,tot,totsqr) default(none)
  for (int i=0;i<nchan;i++) {
    float myamp=0;
    for (int j=0;j<ndata;j++) 
      myamp+=dat[i][j]*tot[j];
    myamp/=totsqr;
    amps[i]=myamp;
    for (int j=0;j<ndata;j++)
      dat[i][j]-=tot[j]*myamp;
  }
  
#ifdef BURST_DUMP_DEBUG
  outfile=fopen("mean_responses2.txt","w");
  for (int i=0;i<nchan;i++)
    fprintf(outfile,"%12.4f\n",amps[i]);
  fclose(outfile);


  {
    float *chansum=vector(nchan);
    float *chansumsqr=vector(nchan);
#pragma omp parallel for
    for (int i=0;i<nchan;i++)
      for (int j=0;j<ndata;j++) {
  chansum[i]+=dat[i][j];
  chansumsqr[i]+=dat[i][j]*dat[i][j];
      }
    outfile=fopen("chan_variances_post.txt","w");
    for (int i=0;i<nchan;i++)
      fprintf(outfile,"%12.6e\n",sqrt(chansumsqr[i]-chansum[i]*chansum[i]/ndata));
    fclose(outfile);
    free(chansum);
    free(chansumsqr);
  }


  #endif
  
  


  free(amps);
  free(tot);
  
}

float get_diagonal_dm(Data *dat) {
  //diagonal DM is when the delay between adjacent channels
  //is equal to the sampling time
  //delay = dm0*dm/nu^2
  // delta delay = dm0*dm*(1//nu1^2 - 1/nu2^2) = dt
  float d1=1.0/dat->chans[0]/dat->chans[0];
  float d2=1.0/dat->chans[1]/dat->chans[1];
  float dm_max=dat->dt/DM0/(d2-d1);
  return dm_max;
}

int get_nchan_from_depth(int depth)
{
  int i=1;
  return i<<depth;
}

float get_diagonal_dm_simple(float nu1, float nu2, float dt, int depth)
{
  float d1=1.0/nu1/nu1;
  float d2=1.0/nu2/nu2;
  int nchan=get_nchan_from_depth(depth);
  //printf("nchan is %d from %d\n",nchan,depth);
  //printf("freqs are %12.4f %12.4f\n",nu1,nu2);
  float dm_max=dt/DM0/( (d2-d1)/nchan);
  //printf("current dm is %12.4f\n",dm_max);
  return fabs(dm_max);
  
}

int get_npass(int n)
{
  int nn=0;
  while (n>1) {
    nn++;
    n/=2;
  }
  return nn;
}

// For allocating output buffer.
size_t burst_get_num_dispersions(size_t nfreq, float freq0,
        float delta_f, int depth) {
  return get_nchan_from_depth(depth);
}

// Return minimum *depth* parameter required to achieve given maximum DM.
int burst_depth_for_max_dm(float max_dm, float delta_t, size_t nfreq, float freq0,
        float delta_f) {
  int depth=2;
  int imax=20;
  int i=0;
  while ((get_diagonal_dm_simple(freq0,freq0+nfreq*delta_f,delta_t,depth)<max_dm)&&(i<imax)) {
    depth++;
    i++;
  }
  if (i==imax) {
    fprintf(stderr,"Failure in burst_depth_for_max_dm.  Did not reach requested DM of %12.4g at a depth of %d\n",max_dm,i);
    return 0;
  }
  
  return depth;
}


//JLS 08 Aug 2014
//since the silly way I have of mapping frequency channels into lambda^2 channels is a bit slow
//but is also static, pre-calculate it and put the mapping into chan_map.
//chan_map should be pre-allocated, with at least 2**depth elements

// If chan_map is an array of indeces, should it not be typed `size_t *`? -KM
void burst_setup_channel_mapping(CM_DTYPE *chan_map, size_t nfreq, float freq0,
        float delta_f, int depth)
{
  int nchan=get_nchan_from_depth(depth);
  int i,j;

  float l0=1.0/(freq0+0.5*delta_f);
  l0=l0*l0;
  float l1=1.0/(freq0+(nfreq-0.5)*delta_f);
  l1=l1*l1;
  if (l0>l1) {
    float tmp=l0;
    l0=l1;
    l1=tmp;
  }

  float *out_chans=(float *)malloc(sizeof(float)*nchan);
  float dl=(l1-l0)/(nchan-1);
  for (i=0;i<nchan;i++)
    out_chans[i]=l0+dl*i;
  
  for (i=0;i<nfreq;i++) {
    float myl=1.0/(freq0+(0.5+i)*delta_f);
    myl=myl*myl;
    int jmin=-1;
    float minerr=1e30;
    for (j=0;j<nchan;j++) {
      float curerr=fabs(myl-out_chans[j]);
      if (curerr<minerr) {
        minerr=curerr;
        jmin=j;
      }
    }
    chan_map[i]=jmin;
  }
  
  
}

/*--------------------------------------------------------------------------------*/
void copy_in_data(Data *dat, float *indata, int ndata)
{
  memset(dat->raw_data[0],0,dat->raw_nchan*dat->ndata*sizeof(float));
  //printf("npad is %d\n",npad);
  
  for (int i=0;i<dat->raw_nchan;i++) {
    for (int j=0;j<ndata;j++) {
      //this line changes depending on memory ordering of input data
      //dat->raw_data[i*dat->ndata+j]=indata1[i*ndata1+j];      
#ifdef BURST_DM_NOTRANSPOSE
      dat->raw_data[i][j]=indata[i*ndata+j];      
#else
      dat->raw_data[i][j]=indata[j*dat->raw_nchan+i];
#endif
    }
  }
}

Data *put_data_into_burst_struct(float *indata, size_t ntime, size_t nfreq, size_t *chan_map, int depth)
{
  
  Data *dat=(Data *)calloc(1,sizeof(Data));
  dat->raw_nchan=nfreq;
  int nchan=get_nchan_from_depth(depth);
  dat->nchan=nchan;
  //int nextra=get_burst_nextra(ntime2,depth);
  dat->ndata=ntime;
  dat->raw_data=matrix(dat->raw_nchan,dat->ndata);
  dat->chan_map=chan_map;
  dat->data=matrix(dat->nchan,dat->ndata);
  copy_in_data(dat,indata,ntime);
  
  return dat;
}

void remap_data(Data *dat)
{					
  assert(dat->chan_map);
  memset(dat->data[0],0,sizeof(dat->data[0][0])*dat->nchan*dat->ndata);  
  for (int i=0;i<dat->raw_nchan;i++) {
    int ii=dat->chan_map[i];
    for (int j=0;j<dat->ndata;j++)
      dat->data[ii][j]+=dat->raw_data[i][j];
  }
}
/*--------------------------------------------------------------------------------*/

void make_rect_mat(float** mat, float* data, int rows, int cols){
	for(int i = 0; i < rows; i++)
		mat[i] = data + i*cols;
}
void make_triangular_mat(float** mat, float* data, int rows, int offset, int delta){
	mat[0] = data;
	for(int i = 1; i < rows; i++){
		mat[i] = &mat[i - 1] + offset + (i - 1)*delta;
	}
}

/*
 * in[j,i] contains the j-th timestream evaluated at time 
 *    t = i - (j+1)s
 * where i=0,...,m-1    j=0,...,n-1    and s is an integer lag
 *
 * The first half of the output array has (length,lag) given by
 *    (m_out, s_out) = (m+s, 2s)
 * and the second half has
 *    (m_out, s_out) = (m+s+1, 2s+1)
 */
void dedisperse_kernel_lagged(float **in, float **out, int n, int m, int s)
{
    // I just haven't thought about the n=odd case
    assert(n % 2 == 0);

    int npair = n/2;
    for (int jj = 0; jj < npair; jj++) {
	for (int i = 0; i < s; i++)
	    out[jj][i] = in[2*jj+1][i];
	for (int i = s; i < m; i++)
	    out[jj][i] = in[2*jj][i-s] + in[2*jj+1][i];
	for (int i = m; i < m+s; i++)
	    out[jj][i] = in[2*jj][i-s];

    
    // for(int i =0; i < n; i++){
    //   printf("dimension, max reached %i x %i, %i\n",n,m,m + i - 1);
    //   printf("in %i, %f\n",i,in[i][m + i - 1]);
    //   printf("ou %i, %f\n",i,out[i][m + i - 1]);
    // }


	for (int i = 0; i < s+1; i++)
	    out[jj+npair][i] = in[2*jj+1][i];

	for (int i = s+1; i < m; i++)
	    out[jj+npair][i] = in[2*jj][i-s-1] + in[2*jj+1][i];
	for (int i = m; i < m+s+1; i++)
	    out[jj+npair][i] = in[2*jj][i-s-1];
      

    }
}

void dedisperse_lagged_partial(float **inin, float**outout, int nchan, int ndat, int npass){
    assert(nchan >= 2);
    assert(ndat >= 1);
    
    // detects underallocation, in the common case where inin was allocated with matrix()
    assert(inin[1] - inin[0] >= nchan + ndat - 1);
    assert(outout[1] - outout[0] >= nchan + ndat - 1);
    
    //assert(nchan == (1 << npass));   // currently require nchan to be a power of two

    int bs = nchan;
    float **in = inin;
    float **out = outout;

    for (int i = 0; i < npass; i++) {    
    #pragma omp parallel for
        for (int j = 0; j < nchan; j += bs)
            dedisperse_kernel_lagged(in+j, out+j, bs, ndat + j/bs, j/bs);

        float **tmp=in;
        in = out;
        out = tmp;
        bs /= 2;
    } 

    // non-rectangular copy
    for (int j = 0; j < nchan; j++)
      memcpy(out[j], in[j], (ndat+j)*sizeof(float));
}


/*
 * Incremental dedispersion is implemented as two steps:
 * dedisperse_lagged() -> update_ring_buffer().
 *
 * @inin is an (nchan, ndat) array
 * @outout is a non-rectangular array: outout[j] points to a buffer of length ndat+j (or larger)
 *
 * WARNING!! Caller must overallocate the 'inin' buffer so that inin[j] has length >= ndat+nchan-1
 * (i.e. inin must be large enough to store the outout array)
 *
 * In current implementation, nchan must be a power of 2, but there is no constraint on ndat.
 */

void dedisperse_lagged(float **inin, float **outout, int nchan, int ndat)
{
    assert(nchan >= 2);
    assert(ndat >= 1);
    
    // detects underallocation, in the common case where inin was allocated with matrix()
    assert(inin[1] - inin[0] >= nchan + ndat - 1);
    assert(outout[1] - outout[0] >= nchan + ndat - 1);
    
    int npass = get_npass(nchan);
    assert(nchan == (1 << npass));   // currently require nchan to be a power of two



    int bs = nchan;
    float **in = inin;
    float **out = outout;

    for (int i = 0; i < npass; i++) {    

  #pragma omp parallel for
      for (int j = 0; j < nchan; j += bs)
          dedisperse_kernel_lagged(in+j, out+j, bs, ndat + j/bs, j/bs);


      float **tmp=in;
      in = out;
      out = tmp;
      bs /= 2;
    } 

    // non-rectangular copy
    for (int j = 0; j < nchan; j++)
      memcpy(out[j], in[j], (ndat+j)*sizeof(float));
}

/*
 * Incremental dedispersion is implemented as two steps:
 * dedisperse_lagged() -> udpate_ring_buffer().
 *
 *   @chunk: This should be the output array from dedisperse_lagged() above.
 *
 *   @nchunk: Should be the same as the @npad argument to dedisperse_lagged().
 *
 *   @ring_buffer: an array of shape (nchan, nring), where nring >= nchan+nchunk.
 *
 *   @ring_t0: this keeps track of the current position in the ring buffer, and
 *       is automatically updated by this routine.  (It can be initialized to zero 
 *       before the first call.)
 *
 * When update_ring_buffer() returns, the following elements of the ring buffer
 * are "valid", i.e. all samples which contribute at the given DM have been summed.
 *
 *     ring_buffer[0][ring_t0-nring:ring_t0]
 *     ring_buffer[1][ring_t0-nring:ring_t0-1]
 *         ...
 *     ring_buffer[nchan][ring_t0-nring:ring_t0-nchan+1]
 *
 * where the index ranges are written in Python notation and are understood to be
 * "wrapped" in the ring buffer.  Invalid elements are in a state of partial summation 
 * and shouldn't be used for anything yet.
 */

void update_ring_buffer(float **chunk, float **ring_buffer, int nchan, int nchunk, int nring, int *ring_t0)
{
    assert(nring >= nchunk + nchan - 1);

    int t0 = *ring_t0;
    //printf("t0 %i\n",t0);

    for (int j = 0; j < nchan; j++) {
#if 0
  // A straightforward implementation using "%" operator turned out to be too slow...
  for (int i = 0; i < j; i++)
      ring_buffer[j][t1+i] += chunk[j][i];   // note += here
  for (int i = j; i < nchunk+j; i++)
      ring_buffer[j][(t0-j+i+nring) % nring] = chunk[j][i];    // note = here
#else
  // ... so I ended up with the following "ugly but fast" implementation instead

  // Update starts at this position in ring buffer
  int t1 = (t0-j+nring) % nring;

  // The logical index range [t1:t1+j] is stored as [t1:t1+n1] and [0:j-n1]   (this defines n1)
  // t2 = position in the ring buffer at the end of this index range
  int n1, t2;
  if (t1+j < nring) {
      n1 = j;
      t2 = t1 + j;
  }
  else {   // wraparound case
      n1 = nring - t1;
      t2 = t1 + j - nring;
  }

  // The logical index range [t2:t2+nchunk] is stored as [t2:t2+n2] and [0:nchunk-n2]
  int n2 = (t2+nchunk < nring) ? nchunk : (nring - t2);

  for (int i = 0; i < n1; i++){
      ring_buffer[j][t1+i] += chunk[j][i];    // note += here
    //*(ring_buffer + j*nring + t1 + i) += *(chunk + j*(nchunk + nchan) + i);
  }
  for (int i = n1; i < j; i++){
      ring_buffer[j][i-n1] += chunk[j][i];    // note += here
    //*(ring_buffer + j*nring + i - n1) += *(chunk + j*(nchunk + nchan) + i);
  }

  for (int i = 0; i < n2; i++){
      ring_buffer[j][t2+i] = chunk[j][j+i];   // note = here
    //*(ring_buffer + j*nring + t2 + i) = *(chunk + j*(nchunk + nchan) + i + j);
  }
  for (int i = n2; i < nchunk; i++){
      ring_buffer[j][i-n2] = chunk[j][j+i];   // note = here 
    //*(ring_buffer + j*nring + i - n2) = *(chunk + j*(nchunk + nchan) + i + j);
  }
#endif
    }

    //*ring_t0 = (t0 + nchunk) % nring;
}
/*--------------------------------------------------------------------------------*/

/* returns data starting at a ringt0 - chunk_size after one call
*   Note that ringt0 corresponds to the value of ringt0 after the call
*   to update_ring_buffer.
*   indata does not have the correct (padded size)
*/

void add_to_ring(DTYPE* indata, DTYPE* outdata, CM_DTYPE* chan_map, DTYPE* ring_buffer_data, int ringt0, int chunk_size, int ring_length, float delta_t, size_t nfreq, float freq0, float delta_f, int depth)
{
     omp_set_dynamic(0);
     omp_set_num_threads(8);

     //zero-pad data
     int ndm = get_nchan_from_depth(depth);
     float * indata_pad = (float*)malloc(sizeof(float)*nfreq*(chunk_size + ndm));
     for(int i = 0; i < nfreq; i++){
       memcpy(indata_pad + i*(chunk_size + ndm), indata + i*chunk_size,sizeof(float)*chunk_size);
       memset(indata_pad + i*(chunk_size + ndm) + chunk_size,0,sizeof(float)*(ndm));
     }


	Data *dat=put_data_into_burst_struct(indata_pad,chunk_size + ndm,nfreq,chan_map,depth);
	remap_data(dat);
     int nchan = dat->nchan;


	float** ring_buffer = (float**)malloc(sizeof(float*)*nchan);
	make_rect_mat(ring_buffer,ring_buffer_data,nchan,ring_length);

	//allocate the triangular matrix for output
	//float* tmp = malloc((nchan*chunk_size + (nchan*(nchan - 1))/2)*sizeof(float));
     //float* tmp = (float*)malloc(nchan*(chunk_size + nchan)*sizeof(float));
	//float** tmp_mat = (float**)malloc(nchan*sizeof(float*));
	//make_triangular_mat(tmp_mat,tmp,nchan,chunk_size,1);
     //make_rect_mat(tmp_mat, tmp, nchan, chunk_size + nchan);
     float** tmp_mat = matrix(nchan,chunk_size + nchan);

	dedisperse_lagged(dat->data,tmp_mat,nchan,chunk_size);
     //printf("ringt0: %i\n",ringt0);
	update_ring_buffer(tmp_mat,ring_buffer,nchan,chunk_size,ring_length,&ringt0);
     //printf("ringt0: %i\n",ringt0);

     //probably not the most efficient way to use the output array
     //does not stop copying if data is incomplete
     //does not prevent overlap
     //ring buffer must be long enough

     //because of the search padding requirement...
	for(int i = 0; i < nchan; i++){
          int src0 = (ring_length + ringt0 - i) % ring_length;
          int src1 = (ring_length + ringt0 + chunk_size - i) % ring_length;
          //printf("ring length %i, cs %i\n",ring_length,chunk_size);
          //printf("i: %i, src0: %i, src1 %i\n",i,src0,src1);
          if (src1 < src0){
           int first_cpy = (ring_length - src0);
           int second_cpy = chunk_size - first_cpy;
           memcpy(outdata + i*(chunk_size + nchan), ring_buffer[i] + src0, first_cpy*sizeof(float));
           memcpy(outdata + i*(chunk_size + nchan) + first_cpy, ring_buffer[i] + src0 + first_cpy, (second_cpy)*sizeof(float));
          }
          else{
		 memcpy(outdata + i*(chunk_size + nchan), ring_buffer[i] + src0, (chunk_size)*sizeof(float));
          }
    }
     free(dat->data[0]);
     free(dat->data);
     free(dat->raw_data[0]);
     free(dat->raw_data);
     free(dat);
	//free(tmp);
     free(indata_pad);
     //free(tmp_mat[0]);
     free(tmp_mat[0]);
     free(tmp_mat);
     //free(ring_buffer);
}

/*--------------------------------------------------------------------------------*/

void find_4567_peaks_wnoise(float *vec, int nsamp, Peak *peak4, Peak *peak5, Peak *peak6, Peak *peak7)
{
  float s4=0,s5=0,s6=0,s7=0;
  float v4=0,v5=0,v6=0,v7=0;
  peak4->ind=0;
  peak5->ind=0;
  peak6->ind=0;
  peak7->ind=0;
  peak4->duration=4;
  peak5->duration=5;
  peak6->duration=6;
  peak7->duration=7;

  float cur4=vec[2]+vec[3]+vec[4]+vec[5];
  peak4->peak=cur4;
  peak5->peak=cur4+vec[6];
  float cur6=vec[0]+vec[1]+cur4;
  peak6->peak=cur6;
  peak7->peak=cur6+vec[6];
  for (int i=6;i<nsamp;i++) {
    cur4=cur4+vec[i];
    s5+=cur4;
    v5+=cur4*cur4;
    if (cur4>peak5->peak) {
      peak5->peak=cur4;
      peak5->ind=i;
    }

    cur4=cur4-vec[i-4];
    s4+=cur4;
    v4+=cur4*cur4;
    if (cur4>peak4->peak) {
      peak4->peak=cur4;
      peak4->ind=i;
    }

    cur6=cur6+vec[i];
    s7+=cur6;
    v7+=cur6*cur6;
    if (cur6>peak7->peak) {
      peak7->peak=cur6;
      peak7->ind=i;
    }

    cur6=cur6-vec[i-6];
    s6+=cur6;
    v6+=cur6*cur6;
    if (cur6>peak6->peak) {
      peak6->peak=cur6;
      peak6->ind=i;
    }

  }
  float n4,n5,n6,n7;
  s4/=(nsamp-7);
  s5/=(nsamp-7);
  s6/=(nsamp-7);
  s7/=(nsamp-7);
  v4/=(nsamp-7);
  v5/=(nsamp-7);
  v6/=(nsamp-7);
  v7/=(nsamp-7);

  n4=sqrt(v4-s4*s4);
  n5=sqrt(v5-s5*s5);
  n6=sqrt(v6-s6*s6);
  n7=sqrt(v7-s7*s7);

  peak4->snr=(peak4->peak-s4)/n4;
  peak5->snr=(peak5->peak-s5)/n5;
  peak6->snr=(peak6->peak-s6)/n6;
  peak7->snr=(peak7->peak-s7)/n7;

  peak4->noise=n4;
  peak5->noise=n5;
  peak6->noise=n6;
  peak7->noise=n7;
}

/*--------------------------------------------------------------------------------*/


Peak find_peaks_wnoise_onedm(float *vec, int nsamples, int max_depth, int cur_depth)
{

  int wt=1<<cur_depth;



  Peak best;  
  best.snr=0;
  best.peak=0;
  //do the 1/2/3 sample case on the first pass through
  if (cur_depth==0) {
    float best1=0;
    float best2=0;
    float best3=0;
    float s1=0,s2=0,s3=0;
    float v1=0,v2=0,v3=0;
    int i1=0,i2=0,i3=0;
    float tmp=vec[0]+vec[1];
    best1=vec[0];
    if (vec[1]>best1)
      best1=vec[1];
    for (int i=2;i<nsamples;i++) {
      if (vec[i]>best1) {
	best1=vec[i];
	i1=i;
      }
      s1+=vec[i];
      v1+=vec[i]*vec[i];
      tmp+=vec[i];
      if (tmp>best3) {
	best3=tmp;
	i3=i;
      }
      s3+=tmp;
      v3+=tmp*tmp;
      tmp-=vec[i-2];
      if (tmp>best2) {
	best2=tmp;
	i2=i;
      }
      s2+=tmp;
      v2+=tmp*tmp;
    }
    s1/=nsamples-2;
    s2/=nsamples-2;
    s3/=nsamples-2;
    v1/=nsamples-2;
    v2/=nsamples-2;
    v3/=nsamples-2;
    v1=sqrt(v1-s1*s1);
    v2=sqrt(v2-s2*s2);
    v3=sqrt(v3-s3*s3);
    float snr1=(best1-s1)/v1;
    float snr2=(best2-s2)/v2;
    float snr3=(best3-s3)/v3;
    if (snr1>best.snr) {
      best.snr=snr1;
      best.peak=best1;
      best.ind=i1;
      best.depth=0;
      best.noise=v1;
      best.duration=1;
    }
    if (snr2>best.snr) {
      best.snr=snr2;
      best.peak=best2;
      best.ind=i2;
      best.depth=0;
      best.noise=v2;
      best.duration=2;
    }
    if (snr3>best.snr) {
      best.snr=snr3;
      best.peak=best3;
      best.ind=i3;
      best.depth=0;
      best.noise=v3;
      best.duration=3;
    }
    
    
    
  }
  
  
  
  Peak peak4,peak5,peak6,peak7;
  find_4567_peaks_wnoise(vec,nsamples,&peak4,&peak5,&peak6,&peak7);
  
  //peak4=peak4/sqrt(4*wt);                                                                                                               
  //peak5=peak5/sqrt(5*wt);                                                                                                               
  //peak6=peak6/sqrt(6*wt);                                                                                                               
  //peak7=peak7/sqrt(7*wt);                                                                                                               





  if (peak4.snr>best.snr)
    best=peak4;
  if (peak5.snr>best.snr)
    best=peak5;
  if (peak6.snr>best.snr)
    best=peak6;
  if (peak7.snr>best.snr)
    best=peak7;
  best.depth=cur_depth;
  //printf("peaks are %12.5g %12.5g %12.5g %12.5g\n",peak4,peak5,peak6,peak7);                                                            

  if (cur_depth<max_depth) {
    int nn=nsamples/2;
    float *vv=(float *)malloc(sizeof(float)*nn);
    for (int i=0;i<nn;i++)
      vv[i]=vec[2*i]+vec[2*i+1];
    Peak new_best=find_peaks_wnoise_onedm(vv,nn,max_depth,cur_depth+1);
    free(vv);
    if (new_best.snr>best.snr)
      best=new_best;
  }
  
  return best;
}

/*--------------------------------------------------------------------------------*/
Peak find_peak_triangle(Data *dat)
{
  //find the longest segment to be searched for
  //can't have a 5-sigma event w/out at least 25 samples to search over
  int max_len=dat->ndata/20;
  int max_seg=max_len/7;
  int max_depth=log2(max_seg);
  //printf("max_depth is %d from %d\n",max_depth,dat->ndata);
  Peak best;
  best.snr=0;
  if (max_depth<1)
    return;
#pragma omp parallel
  {
    Peak mybest;
    mybest.snr=0;
#pragma omp for
    for (int i=0;i<dat->nchan;i++) {
      //This is the only difference,
      //Definitely not code-efficient
      Peak dm_best=find_peaks_wnoise_onedm(dat->data[i],dat->ndata - dat->nchan - i,max_depth,0);
      if (dm_best.snr>mybest.snr) {
  mybest=dm_best;
  mybest.dm_channel=i;
      }
    }
#pragma omp critical
    {
      if (mybest.snr>best.snr)
  best=mybest;
    }
  }
  return best;
}
/*--------------------------------------------------------------------------------*/
Peak find_peak(Data *dat)
{
  //find the longest segment to be searched for
  //can't have a 5-sigma event w/out at least 25 samples to search over
  int max_len=dat->ndata/20;
  int max_seg=max_len/7;
  int max_depth=log2(max_seg);
  //printf("max_depth is %d from %d\n",max_depth,dat->ndata);
  Peak best;
  best.snr=0;
  if (max_depth<1)
    return;
#pragma omp parallel
  {
    Peak mybest;
    mybest.snr=0;
#pragma omp for
    for (int i=0;i<dat->nchan;i++) {
      Peak dm_best=find_peaks_wnoise_onedm(dat->data[i],dat->ndata - dat->nchan,max_depth,0);
      if (dm_best.snr>mybest.snr) {
	mybest=dm_best;
	mybest.dm_channel=i;
      }
    }
#pragma omp critical
    {
      if (mybest.snr>best.snr)
	best=mybest;
    }
  }
  return best;
}
/*--------------------------------------------------------------------------------*/
//Assumes descending triangular format
size_t find_peak_wrapper_triangle(float *data, int nchan, int ndata, float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration)
{
  Data dat;
  float **mat=(float **)malloc(sizeof(float *)*nchan);
  //Assumes a rectangular matrix with a triangular
  //zero fill
  for (int i=0;i<nchan;i++)
    mat[i]=data+i*ndata;
  dat.data=mat;
  dat.ndata=ndata;
  dat.nchan=nchan;
  Peak best=find_peak_triangle(&dat);
  free(dat.data); //get rid of the pointer array
  *peak_snr=best.snr;
  *peak_channel=best.dm_channel;
  //starting sample of the burst
  *peak_sample=best.ind*(1<<best.depth);
  *peak_duration=best.duration*(1<<best.depth);
  return 0;

}
/*--------------------------------------------------------------------------------*/
size_t find_peak_wrapper(float *data, int nchan, int ndata, float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration)
{
  Data dat;
  float **mat=(float **)malloc(sizeof(float *)*nchan);
  for (int i=0;i<nchan;i++)
    mat[i]=data+i*ndata;
  dat.data=mat;
  dat.ndata=ndata;
  dat.nchan=nchan;
  Peak best=find_peak(&dat);
  free(dat.data); //get rid of the pointer array
  *peak_snr=best.snr;
  *peak_channel=best.dm_channel;
  //starting sample of the burst
  *peak_sample=best.ind*(1<<best.depth);
  *peak_duration=best.duration*(1<<best.depth);
  return 0;
  
}