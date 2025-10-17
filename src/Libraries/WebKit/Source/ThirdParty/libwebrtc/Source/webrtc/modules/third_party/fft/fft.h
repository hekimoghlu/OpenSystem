/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
/*--------------------------------*-C-*---------------------------------*
 * File:
 *       fftn.h
 * ---------------------------------------------------------------------*
 * Re[]:        real value array
 * Im[]:        imaginary value array
 * nTotal:      total number of complex values
 * nPass:       number of elements involved in this pass of transform
 * nSpan:       nspan/nPass = number of bytes to increment pointer
 *              in Re[] and Im[]
 * isign: exponent: +1 = forward  -1 = reverse
 * scaling: normalizing constant by which the final result is *divided*
 * scaling == -1, normalize by total dimension of the transform
 * scaling <  -1, normalize by the square-root of the total dimension
 *
 * ----------------------------------------------------------------------
 * See the comments in the code for correct usage!
 */

#ifndef MODULES_THIRD_PARTY_FFT_FFT_H_
#define MODULES_THIRD_PARTY_FFT_FFT_H_

#define FFT_MAXFFTSIZE 2048
#define FFT_NFACTOR 11

typedef struct {
  unsigned int SpaceAlloced;
  unsigned int MaxPermAlloced;
  double Tmp0[FFT_MAXFFTSIZE];
  double Tmp1[FFT_MAXFFTSIZE];
  double Tmp2[FFT_MAXFFTSIZE];
  double Tmp3[FFT_MAXFFTSIZE];
  int Perm[FFT_MAXFFTSIZE];
  int factor[FFT_NFACTOR];

} FFTstr;

/* double precision routine */

int WebRtcIsac_Fftns(unsigned int ndim,
                     const int dims[],
                     double Re[],
                     double Im[],
                     int isign,
                     double scaling,
                     FFTstr* fftstate);

#endif /* MODULES_THIRD_PARTY_FFT_FFT_H_ */
