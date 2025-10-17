/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#if defined(i386) || defined(__arm__)

#include <mach/kern_return.h>
#include <mach/mach_time.h>
#include <stdint.h>

extern void spin_lock(int *);
extern void spin_unlock(int *);

/* deprecated function stub */
kern_return_t
MKGetTimeBaseInfo(
    uint32_t *minAbsoluteTimeDelta,
    uint32_t *theAbsoluteTimeToNanosecondNumerator,
    uint32_t *theAbsoluteTimeToNanosecondDenominator,
    uint32_t *theProcessorToAbsoluteTimeNumerator,
    uint32_t *theProcessorToAbsoluteTimeDenominator
) {
    static struct mach_timebase_info mti = {0};
    static int MKGetTimeBaseInfo_spin_lock = 0;

    if(mti.numer == 0) {
	kern_return_t err;
	spin_lock(&MKGetTimeBaseInfo_spin_lock);
	err = mach_timebase_info(&mti);
	spin_unlock(&MKGetTimeBaseInfo_spin_lock);
	if(err != KERN_SUCCESS)
	    return err;
    }
    if(theAbsoluteTimeToNanosecondNumerator)
	*theAbsoluteTimeToNanosecondNumerator = mti.numer;
    if(theAbsoluteTimeToNanosecondDenominator)
	*theAbsoluteTimeToNanosecondDenominator = mti.denom;
    if(minAbsoluteTimeDelta)
	*minAbsoluteTimeDelta = 1;
    if(theProcessorToAbsoluteTimeNumerator)
	*theProcessorToAbsoluteTimeNumerator = 1;
    if(theProcessorToAbsoluteTimeDenominator)
	*theProcessorToAbsoluteTimeDenominator = 1;
    return KERN_SUCCESS;
}

#endif
