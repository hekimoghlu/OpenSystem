/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#ifndef _PTHREAD_IMPL_H_
#define _PTHREAD_IMPL_H_
/*
 * Internal implementation details
 */

/* This whole header file will disappear, so don't depend on it... */

#if __has_feature(assume_nonnull)
_Pragma("clang assume_nonnull begin")
#endif

#ifndef __POSIX_LIB__

/*
 * [Internal] data structure signatures
 */
#define _PTHREAD_MUTEX_SIG_init		0x32AAABA7

#define _PTHREAD_ERRORCHECK_MUTEX_SIG_init      0x32AAABA1
#define _PTHREAD_RECURSIVE_MUTEX_SIG_init       0x32AAABA2
#define _PTHREAD_FIRSTFIT_MUTEX_SIG_init       0x32AAABA3

#define _PTHREAD_COND_SIG_init		0x3CB0B1BB
#define _PTHREAD_ONCE_SIG_init		0x30B1BCBA
#define _PTHREAD_RWLOCK_SIG_init    0x2DA8B3B4

/*
 * POSIX scheduling policies
 */
#define SCHED_OTHER                1
#define SCHED_FIFO                 4
#define SCHED_RR                   2

#define __SCHED_PARAM_SIZE__       4

#endif /* __POSIX_LIB__ */

#if __has_feature(assume_nonnull)
_Pragma("clang assume_nonnull end")
#endif

#endif /* _PTHREAD_IMPL_H_ */
