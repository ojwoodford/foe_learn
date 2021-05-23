//  G = foe_cd_grad_hmc(J, D, P, step)

#include "mex.h"
#include "matrix.h"
#include <memory.h>
#include <math.h>

template <class A> struct traits {};
template <> struct traits<char>            {enum { type = 0  };};
template <> struct traits<unsigned char>   {enum { type = 1  };};
template <> struct traits<short>           {enum { type = 2  };};
template <> struct traits<unsigned short>  {enum { type = 3  };};
template <> struct traits<long>            {enum { type = 4  };};
template <> struct traits<unsigned long>   {enum { type = 5  };};
template <> struct traits<int>             {enum { type = 6  };};
template <> struct traits<unsigned int>    {enum { type = 7  };};
template <> struct traits<float>           {enum { type = 8  };};
template <> struct traits<double>          {enum { type = 9  };};
#define IS_FP(X) (traits<X>::type == traits<float>::type || traits<X>::type == traits<double>::type)

#include "foe_cd_grad_hmc.h"
#define FILTER_LENGTH (BLOCK_DEPTH*FILTER_WIDTH*FILTER_HEIGHT+(FILTER_OFFSET != 0)+1)
#define BLOCK_PITCH   (BLOCK_WIDTH*BLOCK_HEIGHT*BLOCK_DEPTH)
#define SLIDE_WIDTH   (BLOCK_WIDTH+1-FILTER_WIDTH)
#define SLIDE_HEIGHT  (BLOCK_HEIGHT+1-FILTER_HEIGHT)
#define SLIDE_LENGTH  (SLIDE_WIDTH*SLIDE_HEIGHT)

#define SMALL_DOUBLE       2.2250738585072626e-308
#define LOG_SMALL_DOUBLE   -708.39641853226408

template<class T> static inline double wrapper_func(T *G, const T *J, const void *D, const T *P, const int S_len, const T step, const mxClassID D_class);
template<class T, class U> static inline double foe_cd_grad_hmc(T *G, const T *J, const U *D, const T *P, const int S_len, const T step);
template<class T> static inline void tfilt_grad_x(T *G, const T *J, const T *D);
template<class T> static inline T tfilt_energy_grad_x(T *G, const T *J, const T *D, T *tmp_buf, T *tmp_log);
template<class T> static inline void tfilt_grad_theta(T *G, const T *J, const T *D, const T *P, const T *tmp_buf1, const T *tmp_log1, const T *tmp_buf2, const T *tmp_log2);
template<class T> static inline T dot_prod(const T *F, const T *D, const int offset);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Check number of arguments
	if (nrhs != 4)
		mexErrMsgTxt("4 input arguments expected.");
	if (nlhs != 2)
		mexErrMsgTxt("Unexpected number of output arguments.");

	// Check argument types are valid
	mxClassID var_class = mxGetClassID(prhs[0]);
	if (var_class != mxGetClassID(prhs[2]))
		mexErrMsgTxt("J and P must be of the same class");
	for (int i = 0; i < 4; i++) {
        if (mxIsComplex(prhs[i]))
			mexErrMsgTxt("Inputs cannot be complex.");
	}

	// Get and check array dimensions
	if ((mxGetM(prhs[0]) != FILTER_LENGTH) || (mxGetN(prhs[0]) != NFILTS))
		mexErrMsgTxt("J is of unexpected size.");
	if (mxGetNumberOfElements(prhs[1]) != mxGetNumberOfElements(prhs[2]))
		mexErrMsgTxt("D and P must have the same number of elements");
	if (mxGetNumberOfDimensions(prhs[1]) != 4)
		mexErrMsgTxt("D needs 4 dimensions");
	const mwSize* dim = mxGetDimensions(prhs[1]);
	if ((dim[0] != BLOCK_HEIGHT) || (dim[1] != BLOCK_WIDTH) || (dim[2] != BLOCK_DEPTH))
		mexErrMsgTxt("D has unexpected dimensions");
	if (mxGetNumberOfElements(prhs[3]) != 1)
		mexErrMsgTxt("step must be a scalar");

	// Handle case of D being different class
	mxClassID D_class = mxGetClassID(prhs[1]);
	if (D_class == var_class)
		D_class = mxUNKNOWN_CLASS;

	// Get pointers to input arrays
	const void* J = mxGetData(prhs[0]);
	const void* D = mxGetData(prhs[1]);
	const void* P = mxGetData(prhs[2]);
	double step = mxGetScalar(prhs[3]);
    
	// Create the output array
	plhs[0] = mxCreateNumericMatrix(FILTER_LENGTH, NFILTS, var_class, mxREAL);
	void* G = mxGetData(plhs[0]);
	double rejection_rate;

	// Call the function according to the type
	switch (var_class) {
		case mxDOUBLE_CLASS:
			rejection_rate = wrapper_func((double *)G, (const double *)J, D, (const double*)P, dim[3], step, D_class);
			break;
		case mxSINGLE_CLASS:
			rejection_rate = wrapper_func((float*)G, (const float*)J, D, (const float*)P, dim[3], (float)step, D_class);
			break;
		default:
			mexErrMsgTxt("J and P are of an unsupported type");
			break;
	}
	plhs[1] = mxCreateDoubleScalar(rejection_rate);
	return;
}

template<class T> static inline double wrapper_func(T *G, const T *J, const void *D, const T *P, const int S_len, const T step, const mxClassID D_class)
{
	double rejection_rate;

	if (D_class == mxUNKNOWN_CLASS) {
			rejection_rate = foe_cd_grad_hmc(G, J, (const T *)D, P, S_len, step);
	} else {
		// Call the function according to the type
		switch (D_class) {
			case mxUINT8_CLASS:
				rejection_rate = foe_cd_grad_hmc(G, J, (const unsigned char *)D, P, S_len, step);
				break;
			default:
				mexErrMsgTxt("D is of an unsupported type");
				break;
		}
	}
	return rejection_rate;
}

template<class T, class U> static inline double foe_cd_grad_hmc(T *G, const T *J, const U *D, const T *P, const int S_len, const T step)
{
	T log_offset = log((T)RAND_MAX);

	// Create a buffer to all the data
	T *x_new, *x_old, *grad_x;
	if (traits<T>::type == traits<U>::type) {
		x_new = (T *)mxMalloc(sizeof(T)*(3*BLOCK_PITCH+2*(SLIDE_LENGTH+1)*NFILTS));
		grad_x = &x_new[BLOCK_PITCH];
	} else {
		x_new = (T *)mxMalloc(sizeof(T)*(4*BLOCK_PITCH+2*(SLIDE_LENGTH+1)*NFILTS));
		x_old = &x_new[BLOCK_PITCH];
	    grad_x = &x_old[BLOCK_PITCH];
	}
	T *momentum = &grad_x[BLOCK_PITCH];
	T *tmp_buf1 = &momentum[BLOCK_PITCH];
	T *tmp_log1 = &tmp_buf1[SLIDE_LENGTH*NFILTS];
	T *tmp_buf2 = &tmp_log1[NFILTS];
	T *tmp_log2 = &tmp_buf2[SLIDE_LENGTH*NFILTS];
	
	// Initialize some variables
	int rejections = 0;
	T half_step = step / 2;
	
	// For each block
	for (int s = 0; s < S_len*BLOCK_PITCH; s += BLOCK_PITCH) {		
		// Initialize data and momentum
		if (traits<T>::type == traits<U>::type) {
			x_old = (T *)&D[s];
			memcpy(x_new, x_old, sizeof(T)*BLOCK_PITCH);
		} else {
			// Cast and copy the data all at once
			for (int i = 0; i < BLOCK_PITCH; i++) {
				T tmp_val = (T)D[s+i];
				x_old[i] = tmp_val;
				x_new[i] = tmp_val;
			}
		}
		memcpy(momentum, &P[s], sizeof(T)*BLOCK_PITCH);
		
		// Calculate energy and the gradient w.r.t. data
		T energy = tfilt_energy_grad_x(grad_x, J, x_new, tmp_buf1, tmp_log1);

		// Calculate Hamiltonian energy and make a half step in momentum
		T H = 0;
		for (int i = 0; i < BLOCK_PITCH; i++) {
			H += momentum[i] * momentum[i];
			momentum[i] -= half_step * grad_x[i];
		}
		H /= 2;
		H += energy;

		// Do the required number of steps
		int k = 1;
		while (1) {
			// Update data
			for (int i = 0; i < BLOCK_PITCH; i++)
				x_new[i] += step * momentum[i];

			if (k >= NSTEPS)
				break;
			k++;

			// Calculate the gradient w.r.t. data
			tfilt_grad_x(grad_x, J, x_new);

			// Make a full step in momentum
			for (int i = 0; i < BLOCK_PITCH; i++)
				momentum[i] -= step * grad_x[i];
		}
		
		// Calculate energy and the gradient w.r.t. data
		energy = tfilt_energy_grad_x(grad_x, J, x_new, tmp_buf2, tmp_log2);

		// Make a half step in momentum
		for (int i = 0; i < BLOCK_PITCH; i++)
			momentum[i] -= half_step * grad_x[i];

		// Calculate new Hamiltonian energy
		T Hnew = 0;
		for (int i = 0; i < BLOCK_PITCH; i++)
			Hnew += momentum[i] * momentum[i];
		Hnew /= 2;
		Hnew += energy;
		Hnew -= H;

		// MCMC Metropolis algorithm rejection
		if (Hnew > 0) {
			Hnew += log((T)rand()) - log_offset;
			if (Hnew > 0) {
				rejections++;
				continue; // Rejected
			}
		}

		// Add the gradient of this vector to the overall gradient
		tfilt_grad_theta(G, J, x_old, x_new, tmp_buf1, tmp_log1, tmp_buf2, tmp_log2);
	}
	
	// Free the buffers
	mxFree(x_new);
	
	// Multiply each filter grad by A and divide by the number of data points
	//T r = 1 / (T)S_len;
	T *G_ptr = G;
	for (int f = 0; f < NFILTS; f++) {
		//T a = r * J[f*FILTER_LENGTH+FILTER_LENGTH-1];
		T a = J[f*FILTER_LENGTH+FILTER_LENGTH-1];
		// For each coefficient
		for (int i = 0; i < FILTER_LENGTH; i++) 
			(*G_ptr++) *= a;
	}

	// Return the rejection rate
	//return (double)rejections / (double)S_len;
	return (double)rejections;
}

template<class T> static inline void tfilt_grad_x(T *G, const T *J, const T *D)
{
	// Reset the gradient to 0
	memset(G, 0, BLOCK_PITCH*sizeof(T));

	// For each filter
	for (int f = 0; f < FILTER_LENGTH*NFILTS; f += FILTER_LENGTH) {
		const T *F;
		
		// For each vector
		for (int x = 0; x < SLIDE_WIDTH*BLOCK_HEIGHT; x += BLOCK_HEIGHT) {
			for (int y = 0; y < SLIDE_HEIGHT; y++) {
				int offset = x + y;
				// Dot product
				T dot_d = dot_prod(&J[f], D, offset);

				// Calculate the gradient w.r.t. the data
				dot_d /= dot_d * dot_d * (T)0.5 + 1;
				dot_d *= J[f+(FILTER_LENGTH-1)];
				F = &J[f+FILTER_OFFSET];
				for (int p = 0; p < BLOCK_PITCH; p += BLOCK_WIDTH*BLOCK_HEIGHT) {
					T *G_ptr = &G[p+offset];
					for (int n = 0; n < FILTER_WIDTH*BLOCK_HEIGHT; n += BLOCK_HEIGHT) {
						for (int m = 0; m < FILTER_HEIGHT; m++) {
							G_ptr[m+n] += dot_d * *F++;
						}
					}
				}
			}
		}
	}
	return;
}

template<class T> static inline T tfilt_energy_grad_x(T *G, const T *J, const T *D, T *tmp_buf, T *tmp_log)
{
	T energy = 0;

	// Reset the gradient to 0
	memset(G, 0, BLOCK_PITCH*sizeof(T));

	// For each filter
	for (int f = 0; f < FILTER_LENGTH*NFILTS; f += FILTER_LENGTH) {
		double prod_prelog = SMALL_DOUBLE;
		const T *F;
		
		// For each vector
		for (int x = 0; x < SLIDE_WIDTH*BLOCK_HEIGHT; x += BLOCK_HEIGHT) {
			for (int y = 0; y < SLIDE_HEIGHT; y++) {
				int offset = x + y;
				// Dot product
				T dot_d = dot_prod(&J[f], D, offset);
		
				// Calculate square of the dot product, plus some...
				T sq_d = dot_d * dot_d * (T)0.5 + 1;
				prod_prelog *= (double)sq_d;

				// Store this value as it will be required if we calculate gradient w.r.t. parameters
				dot_d /= sq_d;
				*tmp_buf++ = dot_d;

				// Calculate the gradient w.r.t. the data
				dot_d *= J[f+(FILTER_LENGTH-1)];
				F = &J[f+FILTER_OFFSET];
				for (int p = 0; p < BLOCK_PITCH; p += BLOCK_WIDTH*BLOCK_HEIGHT) {
					T *G_ptr = &G[p+offset];
					for (int n = 0; n < FILTER_WIDTH*BLOCK_HEIGHT; n += BLOCK_HEIGHT) {
						for (int m = 0; m < FILTER_HEIGHT; m++) {
							G_ptr[m+n] += dot_d * *F++;
						}
					}
				}
			}
		}
		T logsum = (T)(log(prod_prelog) - LOG_SMALL_DOUBLE);
		// Store this value as it will be required if we calculate gradient w.r.t. parameters
		*tmp_log++ = logsum;
		// Calculate energy
		energy += *F * logsum;
	}

	return energy;
}

template<class T> static inline void tfilt_grad_theta(T *G, const T *J, const T *D, const T *P, const T *tmp_buf1, const T *tmp_log1, const T *tmp_buf2, const T *tmp_log2)
{
	// For each filter
	for (int f = 0; f < NFILTS; f++) {

		if (FILTER_OFFSET) {
			// For the filter offset just sum values
			int k = f * SLIDE_LENGTH;
			T sum = 0;
			// For each vector
			for (int v = 0; v < SLIDE_LENGTH; v++)
				sum += tmp_buf2[k+v] - tmp_buf1[k+v];
			(*G++) += sum;
		}

		// For the other filter values, multiply by the respective value in each
		// vector before summing.
		unsigned long offset = 0;
		for (int p = 0; p < BLOCK_DEPTH; p++) {
			for (int n = 0; n < FILTER_WIDTH; n++) {
				for (int m = 0; m < FILTER_HEIGHT; m++) {
					int k = f * SLIDE_LENGTH;
					T sum = 0;
					// For each vector
					for (int x = 0; x < SLIDE_WIDTH*BLOCK_HEIGHT; x += BLOCK_HEIGHT) {
						for (int y = 0; y < SLIDE_HEIGHT; y++)
							sum += P[offset+x+y] * tmp_buf2[k+y] - D[offset+x+y] * tmp_buf1[k+y];
						k += SLIDE_HEIGHT;
					}
					(*G++) += sum;
					offset++;
				}
				offset += (SLIDE_HEIGHT - 1);
			}
			offset += (SLIDE_WIDTH - 1) * BLOCK_HEIGHT;
		}

		// Add on logsum to alpha
		(*G++) += tmp_log2[f] - tmp_log1[f];
	}		
}

template<class T> static inline T dot_prod(const T *F, const T *D, const int offset)
{
	T dot_d;
	if (FILTER_OFFSET)
		dot_d = *F++; // Add on filter offsets
	else
		dot_d = 0;

	// Dot product
	for (int p = 0; p < BLOCK_PITCH; p += BLOCK_WIDTH*BLOCK_HEIGHT) {
		const T *D_ptr = &D[p+offset];
		for (int n = 0; n < FILTER_WIDTH*BLOCK_HEIGHT; n += BLOCK_HEIGHT) {
			for (int m = 0; m < FILTER_HEIGHT; m++) {
				dot_d += D_ptr[m+n] * *F++;
			}
		}
	}
	return dot_d;
}
