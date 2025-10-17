/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#ifndef _MACH_MACH_TIME_PRIVATE_H_
#define _MACH_MACH_TIME_PRIVATE_H_

#include <mach/mach_types.h>
#include <sys/cdefs.h>
#include <Availability.h>

__BEGIN_DECLS
#ifndef KERNEL
// Forward definition because this is a BSD value
struct timespec;

__OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0)
kern_return_t           mach_get_times(uint64_t* absolute_time,
    uint64_t* continuous_time,
    struct timespec *tp);

__OSX_AVAILABLE(10.12) __IOS_AVAILABLE(10.0) __TVOS_AVAILABLE(10.0) __WATCHOS_AVAILABLE(3.0)
uint64_t                mach_boottime_usec(void);

#endif /* KERNEL */

__END_DECLS

#endif /* _MACH_MACH_TIME_PRIVATE_H_ */
