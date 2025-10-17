/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#ifndef _MACH_MACH_TIME_H_
#define _MACH_MACH_TIME_H_

#include <mach/mach_types.h>
#include <sys/cdefs.h>
#include <Availability.h>

struct mach_timebase_info {
	uint32_t        numer;
	uint32_t        denom;
};

typedef struct mach_timebase_info       *mach_timebase_info_t;
typedef struct mach_timebase_info       mach_timebase_info_data_t;

__BEGIN_DECLS
#ifndef KERNEL

kern_return_t           mach_timebase_info(
	mach_timebase_info_t    info);

kern_return_t           mach_wait_until(
	uint64_t                deadline);

#endif  /* KERNEL */

uint64_t                        mach_absolute_time(void);

#ifndef KERNEL
__OSX_AVAILABLE_STARTING(__MAC_10_10, __IPHONE_8_0)
#endif
uint64_t                        mach_approximate_time(void);

/*
 * like mach_absolute_time, but advances during sleep
 */
#ifndef KERNEL
__OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0)
#endif
uint64_t                        mach_continuous_time(void);

/*
 * like mach_approximate_time, but advances during sleep
 */
#ifndef KERNEL
__OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0)
#endif
uint64_t                        mach_continuous_approximate_time(void);

/*
 * variant of mach_continuous_time that uses speculative timebase
 */
#ifdef KERNEL
uint64_t                        mach_continuous_speculative_time(void);
#endif

__END_DECLS

#ifdef PRIVATE
#include <mach/mach_time_private.h>
#endif

#endif /* _MACH_MACH_TIME_H_ */
