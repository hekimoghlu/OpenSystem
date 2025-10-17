/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#ifndef __OS_ALLOC__
#define __OS_ALLOC__

#ifndef __OS_ALLOC_INDIRECT__
#define __OS_ALLOC_INDIRECT__
#endif // __OS_ALLOC_INDIRECT__

#include <os/alloc_once_impl.h>

/* Keys for use with os_once_alloc. */

/* Keys for Libsystem. */
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_NOTIFY			0
#define OS_ALLOC_ONCE_KEY_LIBXPC					1
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_C				2
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_INFO			3
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_NETWORK			4
#define OS_ALLOC_ONCE_KEY_LIBCACHE					5
#define OS_ALLOC_ONCE_KEY_LIBCOMMONCRYPTO			6
#define OS_ALLOC_ONCE_KEY_LIBDISPATCH				7
#define OS_ALLOC_ONCE_KEY_LIBDYLD					8
#define OS_ALLOC_ONCE_KEY_LIBKEYMGR					9
#define OS_ALLOC_ONCE_KEY_LIBLAUNCH					10
#define OS_ALLOC_ONCE_KEY_LIBMACHO					11
#define OS_ALLOC_ONCE_KEY_OS_TRACE					12
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_BLOCKS			13
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_MALLOC			14
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_PLATFORM		15
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_PTHREAD			16
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_STATS			17
#define OS_ALLOC_ONCE_KEY_LIBSECINIT				18
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_CORESERVICES	19
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_SYMPTOMS		20
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_PLATFORM_ASL	21
#define OS_ALLOC_ONCE_KEY_LIBSYSTEM_FEATUREFLAGS	22

/* Keys OS_ALLOC_ONCE_KEY_MAX - 10 upwards are reserved for the system. */
#define OS_ALLOC_ONCE_KEY_RESERVED_0	(OS_ALLOC_ONCE_KEY_MAX - 10)
#define OS_ALLOC_ONCE_KEY_RESERVED_1	(OS_ALLOC_ONCE_KEY_MAX - 9)
#define OS_ALLOC_ONCE_KEY_RESERVED_2	(OS_ALLOC_ONCE_KEY_MAX - 8)
#define OS_ALLOC_ONCE_KEY_RESERVED_3	(OS_ALLOC_ONCE_KEY_MAX - 7)
#define OS_ALLOC_ONCE_KEY_RESERVED_4	(OS_ALLOC_ONCE_KEY_MAX - 6)
#define OS_ALLOC_ONCE_KEY_RESERVED_5	(OS_ALLOC_ONCE_KEY_MAX - 5)
#define OS_ALLOC_ONCE_KEY_RESERVED_6	(OS_ALLOC_ONCE_KEY_MAX - 4)
#define OS_ALLOC_ONCE_KEY_RESERVED_7	(OS_ALLOC_ONCE_KEY_MAX - 3)
#define OS_ALLOC_ONCE_KEY_RESERVED_8	(OS_ALLOC_ONCE_KEY_MAX - 2)
#define OS_ALLOC_ONCE_KEY_RESERVED_9	(OS_ALLOC_ONCE_KEY_MAX - 1)

/* OS_ALLOC_ONCE_KEY_MAX cannot be used. */

#endif // __OS_ALLOC__
