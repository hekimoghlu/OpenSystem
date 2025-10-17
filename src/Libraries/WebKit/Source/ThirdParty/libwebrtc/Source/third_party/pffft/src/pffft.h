/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/*
   PFFFT : a Pretty Fast FFT.

   This is basically an adaptation of the single precision fftpack
   (v4) as found on netlib taking advantage of SIMD instruction found
   on cpus such as intel x86 (SSE1), powerpc (Altivec), and arm (NEON).
   
   For architectures where no SIMD instruction is available, the code
   falls back to a scalar version.  

   Restrictions: 

   - 1D transforms only, with 32-bit single precision.

   - supports only transforms for inputs of length N of the form
   N=(2^a)*(3^b)*(5^c), a >= 5, b >=0, c >= 0 (32, 48, 64, 96, 128,
   144, 160, etc are all acceptable lengths). Performance is best for
   128<=N<=8192.

   - all (float*) pointers in the functions below are expected to
   have an "simd-compatible" alignment, that is 16 bytes on x86 and
   powerpc CPUs.
  
   You can allocate such buffers with the functions
   pffft_aligned_malloc / pffft_aligned_free (or with stuff like
   posix_memalign..)

*/

#ifndef PFFFT_H
#define PFFFT_H

#include <stddef.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PFFFT_SIMD_DISABLE
// Detects compiler bugs with respect to simd instruction.
void validate_pffft_simd(void);
#endif

/* opaque struct holding internal stuff (precomputed twiddle factors)
   this struct can be shared by many threads as it contains only
   read-only data.
*/
typedef struct PFFFT_Setup PFFFT_Setup;

/* direction of the transform */
typedef enum { PFFFT_FORWARD, PFFFT_BACKWARD } pffft_direction_t;

/* type of transform */
typedef enum { PFFFT_REAL, PFFFT_COMPLEX } pffft_transform_t;

/*
  prepare for performing transforms of size N -- the returned
  PFFFT_Setup structure is read-only so it can safely be shared by
  multiple concurrent threads.
*/
PFFFT_Setup* pffft_new_setup(int N, pffft_transform_t transform);
void pffft_destroy_setup(PFFFT_Setup*);
/*
   Perform a Fourier transform , The z-domain data is stored in the
   most efficient order for transforming it back, or using it for
   convolution. If you need to have its content sorted in the
   "usual" way, that is as an array of interleaved complex numbers,
   either use pffft_transform_ordered , or call pffft_zreorder after
   the forward fft, and before the backward fft.

   Transforms are not scaled: PFFFT_BACKWARD(PFFFT_FORWARD(x)) = N*x.
   Typically you will want to scale the backward transform by 1/N.

   The 'work' pointer should point to an area of N (2*N for complex
   fft) floats, properly aligned. If 'work' is NULL, then stack will
   be used instead (this is probably the best strategy for small
   FFTs, say for N < 16384).

   input and output may alias.
*/
void pffft_transform(PFFFT_Setup* setup,
                     const float* input,
                     float* output,
                     float* work,
                     pffft_direction_t direction);

/*
   Similar to pffft_transform, but makes sure that the output is
   ordered as expected (interleaved complex numbers).  This is
   similar to calling pffft_transform and then pffft_zreorder.

   input and output may alias.
*/
void pffft_transform_ordered(PFFFT_Setup* setup,
                             const float* input,
                             float* output,
                             float* work,
                             pffft_direction_t direction);

/*
   call pffft_zreorder(.., PFFFT_FORWARD) after pffft_transform(...,
   PFFFT_FORWARD) if you want to have the frequency components in
   the correct "canonical" order, as interleaved complex numbers.

   (for real transforms, both 0-frequency and half frequency
   components, which are real, are assembled in the first entry as
   F(0)+i*F(n/2+1). Note that the original fftpack did place
   F(n/2+1) at the end of the arrays).

   input and output should not alias.
*/
void pffft_zreorder(PFFFT_Setup* setup,
                    const float* input,
                    float* output,
                    pffft_direction_t direction);

/*
   Perform a multiplication of the frequency components of dft_a and
   dft_b and accumulate them into dft_ab. The arrays should have
   been obtained with pffft_transform(.., PFFFT_FORWARD) and should
   *not* have been reordered with pffft_zreorder (otherwise just
   perform the operation yourself as the dft coefs are stored as
   interleaved complex numbers).

   the operation performed is: dft_ab += (dft_a * fdt_b)*scaling

   The dft_a, dft_b and dft_ab pointers may alias.
*/
void pffft_zconvolve_accumulate(PFFFT_Setup* setup,
                                const float* dft_a,
                                const float* dft_b,
                                float* dft_ab,
                                float scaling);

/*
  the float buffers must have the correct alignment (16-byte boundary
  on intel and powerpc). This function may be used to obtain such
  correctly aligned buffers.
*/
void* pffft_aligned_malloc(size_t nb_bytes);
void pffft_aligned_free(void*);

/* return 4 or 1 wether support SSE/Altivec instructions was enable when
 * building pffft.c */
int pffft_simd_size(void);

#ifdef __cplusplus
}
#endif

#endif // PFFFT_H
