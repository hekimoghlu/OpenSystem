/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#ifndef _SCHED_H_
#define _SCHED_H_

#include <sys/cdefs.h>
#include <pthread/pthread_impl.h>

__BEGIN_DECLS
/*
 * Scheduling paramters
 */
#ifndef __POSIX_LIB__
struct sched_param { int sched_priority;  char __opaque[__SCHED_PARAM_SIZE__]; };
#else
struct sched_param;
#endif

extern int sched_yield(void);
extern int sched_get_priority_min(int);
extern int sched_get_priority_max(int);
__END_DECLS

#endif /* _SCHED_H_ */

