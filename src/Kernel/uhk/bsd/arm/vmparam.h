/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#ifndef _BSD_ARM_VMPARAM_H_
#define _BSD_ARM_VMPARAM_H_ 1

#if defined (__arm__) || defined (__arm64__)

#include <sys/resource.h>

#ifndef KERNEL
#include <TargetConditionals.h>
#endif

#define USRSTACK        (0x27E00000)    /* ASLR slides stack down by up to 1MB */
#define USRSTACK64      (0x000000016FE00000ULL)

/*
 * Virtual memory related constants, all in bytes
 */
#ifndef DFLDSIZ
#define DFLDSIZ         (RLIM_INFINITY)         /* initial data size limit */
#endif
#ifndef MAXDSIZ
#define MAXDSIZ         (RLIM_INFINITY)         /* max data size */
#endif
#ifndef DFLSSIZ
/* XXX stack size default is a platform property: use getrlimit(2) */
#if (defined(TARGET_OS_OSX) && (TARGET_OS_OSX != 0)) || \
        (defined(KERNEL) && XNU_TARGET_OS_OSX)
#define DFLSSIZ         (8*1024*1024 - 16*1024)
#else
#define DFLSSIZ         (1024*1024 - 16*1024)   /* initial stack size limit */
#endif /* TARGET_OS_OSX .. || XNU_KERNEL_PRIVATE .. */
#endif /* DFLSSIZ */
#ifndef MAXSSIZ
/* XXX stack size limit is a platform property: use getrlimit(2) */
#if (defined(TARGET_OS_OSX) && (TARGET_OS_OSX != 0)) || \
        (defined(KERNEL) && XNU_TARGET_OS_OSX)
#define MAXSSIZ         (64*1024*1024)          /* max stack size */
#else
#define MAXSSIZ         (1024*1024)             /* max stack size */
#endif /* TARGET_OS_OSX .. || XNU_KERNEL_PRIVATE .. */
#endif /* MAXSSIZ */
#ifndef DFLCSIZ
#define DFLCSIZ         (0)                     /* initial core size limit */
#endif
#ifndef MAXCSIZ
#define MAXCSIZ         (RLIM_INFINITY)         /* max core size */
#endif  /* MAXCSIZ */

#endif /* defined (__arm__) || defined (__arm64__) */

#endif  /* _BSD_ARM_VMPARAM_H_ */
