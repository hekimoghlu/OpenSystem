/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */

#ifndef _OS_OSBASE_H
#define _OS_OSBASE_H

#include <sys/cdefs.h>

#include <libkern/OSTypes.h>

#include <stdint.h>

__BEGIN_DECLS

#ifdef  KERNEL_PRIVATE

OS_INLINE
uint64_t
__OSAbsoluteTime(
	AbsoluteTime    abstime)
{
	return *(uint64_t *)&abstime;
}

OS_INLINE
uint64_t *
__OSAbsoluteTimePtr(
	AbsoluteTime    *abstime)
{
	return (uint64_t *)abstime;
}

#define AbsoluteTime_to_scalar(x)       (*(uint64_t *)(x))

/* t1 < = > t2 */
#define CMP_ABSOLUTETIME(t1, t2)                                \
	(AbsoluteTime_to_scalar(t1) >                           \
	        AbsoluteTime_to_scalar(t2)? (int)+1 :   \
	 (AbsoluteTime_to_scalar(t1) <                          \
	        AbsoluteTime_to_scalar(t2)? (int)-1 : 0))

/* t1 += t2 */
#define ADD_ABSOLUTETIME(t1, t2)                                \
	(AbsoluteTime_to_scalar(t1) +=                          \
	                        AbsoluteTime_to_scalar(t2))

/* t1 -= t2 */
#define SUB_ABSOLUTETIME(t1, t2)                                \
	(AbsoluteTime_to_scalar(t1) -=                          \
	                        AbsoluteTime_to_scalar(t2))

#define ADD_ABSOLUTETIME_TICKS(t1, ticks)               \
	(AbsoluteTime_to_scalar(t1) +=                          \
	                                        (int32_t)(ticks))

#endif  /* KERNEL_PRIVATE */

__END_DECLS

#endif /* _OS_OSBASE_H */
