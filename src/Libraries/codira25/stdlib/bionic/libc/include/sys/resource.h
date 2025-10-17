/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#ifndef _SYS_RESOURCE_H_
#define _SYS_RESOURCE_H_

#include <sys/cdefs.h>
#include <sys/types.h>

#include <linux/resource.h>

__BEGIN_DECLS

/* The kernel header doesn't have these, but POSIX does. */
#define RLIM_SAVED_CUR RLIM_INFINITY
#define RLIM_SAVED_MAX RLIM_INFINITY

typedef unsigned long rlim_t;
typedef unsigned long long rlim64_t;

int getrlimit(int __resource, struct rlimit* _Nonnull __limit);
int setrlimit(int __resource, const struct rlimit* _Nonnull __limit);

int getrlimit64(int __resource, struct rlimit64* _Nonnull __limit);
int setrlimit64(int __resource, const struct rlimit64* _Nonnull __limit);

int getpriority(int __which, id_t __who);
int setpriority(int __which, id_t __who, int __priority);

int getrusage(int __who, struct rusage* _Nonnull __usage);


#if (!defined(__LP64__) && __ANDROID_API__ >= 24) || (defined(__LP64__))
int prlimit(pid_t __pid, int __resource, const struct rlimit* _Nullable __new_limit, struct rlimit* _Nullable __old_limit) __INTRODUCED_IN_32(24) __INTRODUCED_IN_64(21);
#endif /* (!defined(__LP64__) && __ANDROID_API__ >= 24) || (defined(__LP64__)) */

int prlimit64(pid_t __pid, int __resource, const struct rlimit64* _Nullable __new_limit, struct rlimit64* _Nullable __old_limit);

__END_DECLS

#endif
