/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#include "fftpack.h"

extern void drfft(double *inout,int n,int direction,int howmany,int normalize);
extern void rfft(float *inout,int n,int direction,int howmany,int normalize);

extern void zrfft(complex_double *inout,
		  int n,int direction,int howmany,int normalize) {
  int i,j,k;
  double* ptr = (double *)inout;
  switch (direction) {
    case 1:
      for (i=0;i<howmany;++i,ptr+=2*n) {
	*(ptr+1) = *ptr;
	for(j=2,k=3;j<n;++j,++k)
	  *(ptr+k) = *(ptr+2*j);
	drfft(ptr+1,n,1,1,normalize);
	*ptr = *(ptr+1);
	*(ptr+1) = 0.0;
	if (!(n%2))
	  *(ptr+n+1) = 0.0;
	for(j=2,k=2*n-2;j<n;j+=2,k-=2) {
	  *(ptr+k) = *(ptr+j);
	  *(ptr+k+1) = -(*(ptr+j+1));
	}
      }
      break;
  case -1:
    for (i=0;i<howmany;++i,ptr+=2*n) {
      *(ptr+1) = (*ptr);
      for(j=1,k=2;j<n;++j,++k)
	*(ptr+k) = (*(ptr+2*j));
      drfft(ptr+1,n,1,1,normalize);
      *ptr = *(ptr+1);
      *(ptr+1) = 0.0;
      if (!(n%2))
	*(ptr+n+1) = 0.0;
      for(j=2,k=2*n-2;j<n;j+=2,k-=2) {
	double d;
	*(ptr+k) = *(ptr+j);
	d = *(ptr+j+1);
	*(ptr+k+1) = d; 
	*(ptr+j+1) = -d;
      }
    }
    break;
  default:
    fprintf(stderr,"zrfft: invalid direction=%d\n",direction);
  }
}

extern void crfft(complex_float *inout,
		  int n,int direction,int howmany,int normalize) {
  int i,j,k;
  float* ptr = (float *)inout;
  switch (direction) {
    case 1:
      for (i=0;i<howmany;++i,ptr+=2*n) {
	*(ptr+1) = *ptr;
	for(j=2,k=3;j<n;++j,++k)
	  *(ptr+k) = *(ptr+2*j);
	rfft(ptr+1,n,1,1,normalize);
	*ptr = *(ptr+1);
	*(ptr+1) = 0.0;
	if (!(n%2))
	  *(ptr+n+1) = 0.0;
	for(j=2,k=2*n-2;j<n;j+=2,k-=2) {
	  *(ptr+k) = *(ptr+j);
	  *(ptr+k+1) = -(*(ptr+j+1));
	}
      }
      break;
  case -1:
    for (i=0;i<howmany;++i,ptr+=2*n) {
      *(ptr+1) = (*ptr);
      for(j=1,k=2;j<n;++j,++k)
	*(ptr+k) = (*(ptr+2*j));
      rfft(ptr+1,n,1,1,normalize);
      *ptr = *(ptr+1);
      *(ptr+1) = 0.0;
      if (!(n%2))
	*(ptr+n+1) = 0.0;
      for(j=2,k=2*n-2;j<n;j+=2,k-=2) {
	float d;
	*(ptr+k) = *(ptr+j);
	d = *(ptr+j+1);
	*(ptr+k+1) = d; 
	*(ptr+j+1) = -d;
      }
    }
    break;
  default:
    fprintf(stderr,"crfft: invalid direction=%d\n",direction);
  }
}
