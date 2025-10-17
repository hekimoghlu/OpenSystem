/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
/* Copyright (c) 1998,2011,2014 Apple Inc.  All Rights Reserved.
 *
 * NOTICE: USE OF THE MATERIALS ACCOMPANYING THIS NOTICE IS SUBJECT
 * TO THE TERMS OF THE SIGNED "FAST ELLIPTIC ENCRYPTION (FEE) REFERENCE
 * SOURCE CODE EVALUATION AGREEMENT" BETWEEN APPLE, INC. AND THE
 * ORIGINAL LICENSEE THAT OBTAINED THESE MATERIALS FROM APPLE,
 * INC.  ANY USE OF THESE MATERIALS NOT PERMITTED BY SUCH AGREEMENT WILL
 * EXPOSE YOU TO LIABILITY.
 ***************************************************************************
 *
 * Measurement of feemods and mulgs withing an elliptic_simple() call.
 */

#include "feeDebug.h"

#ifdef	FEE_DEBUG
#define ELLIPTIC_MEASURE	0
#else	// FEE_DEBUG
#define ELLIPTIC_MEASURE	0	/* always off */
#endif	// FEE_DEBUG

#if	ELLIPTIC_MEASURE

extern int doEllMeasure;	// gather stats on/off */
extern int bitsInN;
extern int numFeeMods;
extern int numMulgs;

#define START_ELL_MEASURE(n)		\
	doEllMeasure = 1;		\
	bitsInN = bitlen(n);		\
	numFeeMods = 0;			\
	numMulgs = 0;

#define END_ELL_MEASURE		doEllMeasure = 0;

#define INCR_FEEMODS			\
	if(doEllMeasure) {		\
		numFeeMods++;		\
	}

#define INCR_MULGS			\
	if(doEllMeasure) {		\
		numMulgs++;		\
	}

/*
 * These two are used around mulg() calls in feemod() itself; they
 * inhibit the counting of those mulg() calls.
 */
#define PAUSE_ELL_MEASURE				\
	{						\
		int tempEllMeasure = doEllMeasure;	\
		doEllMeasure = 0;

#define RESUME_ELL_MEASURE				\
		doEllMeasure = tempEllMeasure;		\
	}

#else	// ELLIPTIC_MEASURE

#define START_ELL_MEASURE(n)
#define END_ELL_MEASURE
#define INCR_FEEMODS
#define INCR_MULGS
#define PAUSE_ELL_MEASURE
#define RESUME_ELL_MEASURE

#endif	// ELLIPTIC_MEASURE
