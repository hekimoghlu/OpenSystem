/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
 */

#ifndef	_CK_ECDSA_PROFILE_H_
#define _CK_ECDSA_PROFILE_H_

#include "ckconfig.h"

#include "feeDebug.h"

#ifdef	FEE_DEBUG
#define ECDSA_PROFILE	0
#else	/* FEE_DEBUG */
#define ECDSA_PROFILE	0	/* always off */
#endif	/* FEE_DEBUG */

#if	ECDSA_PROFILE

#include <kern/time_stamp.h>

/*
 * Unlike the profiling macros in feeDebug.h, these are intended to
 * be used for fragments of code, not entire functions.
 */
#define SIGPROF_START 				\
{						\
	struct tsval _profStartTime;		\
	struct tsval _profEndTime;		\
	kern_timestamp(&_profStartTime);

/*
 * This one goes at the end of the routine, just before the (only) return.
 * There must be a static accumulator (an unsigned int) on a per-routine basis.
 */
#define SIGPROF_END(accum)						\
	kern_timestamp(&_profEndTime);					\
	accum += (_profEndTime.low_val - _profStartTime.low_val);	\
}


/*
 * Accumulators.
 */
extern unsigned signStep1;
extern unsigned signStep2;
extern unsigned signStep34;
extern unsigned signStep5;
extern unsigned signStep67;
extern unsigned signStep8;
extern unsigned vfyStep1;
extern unsigned vfyStep3;
extern unsigned vfyStep4;
extern unsigned vfyStep5;
extern unsigned vfyStep6;
extern unsigned vfyStep7;

#else	/* ECDSA_PROFILE */

#define SIGPROF_START
#define SIGPROF_END(accum)

#endif	/* ECDSA_PROFILE */

#endif	/* _CK_ECDSA_PROFILE_H_ */
