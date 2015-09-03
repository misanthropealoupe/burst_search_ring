#ifndef RING_BUFFER_H__
#define RING_BUFFER_H__

#define CM_DTYPE size_t
//#define CM_DTYPE int64_t
#define DTYPE float

typedef struct {
  //float dm_max;
  //float dm_offset;
  int nchan;
  int raw_nchan;
  int ndata;
  float *chans;  //delay in pixels is chan_map[i]*dm
  float *raw_chans; 
  float dt;

  float **raw_data; //remap to float so we can do things like clean up data without worrying about overflow
  float **data;
  size_t *chan_map;

  //int icur;  //useful if we want to collapse the data after dedispersing
} Data;

/*--------------------------------------------------------------------------------*/

typedef struct {
  float snr;
  float peak;
  int ind;
  int depth;
  float noise;
  int dm_channel;
  int duration;
  long global_ind;
} Peak;

int get_nchan_from_depth(int depth);
float get_diagonal_dm_simple(float nu1, float nu2, float dt, int depth);
//void dedisperse_lagged_partial(float **inin, float**outout, int nchan, int ndat, int npass);
Data *put_data_into_burst_struct(float *indata, size_t ntime, size_t nfreq, size_t *chan_map, int depth);
size_t find_peak_wrapper(float *data, int nchan, int ndata, float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration);
size_t find_peak_wrapper_triangle(float *data, int nchan, int ndata, float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration);
size_t burst_get_num_dispersions(size_t nfreq, float freq0,float delta_f, int depth);
void clean_rows_2pass(float *vec, size_t nchan, size_t ndata);
int burst_depth_for_max_dm(float max_dm, float delta_t, size_t nfreq, float freq0,float delta_f);
void add_to_ring(DTYPE* indata, DTYPE* outdata, CM_DTYPE* chan_map, DTYPE* ring_buffer_data, int ringt0, int chunk_size, int ring_length, float delta_t, size_t nfreq, float freq0, float delta_f, int depth);
void burst_setup_channel_mapping(CM_DTYPE *chan_map, size_t nfreq, float freq0, float delta_f, int depth);
#endif