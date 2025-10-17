/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#ifndef __PLATFORM_H
#define __PLATFORM_H

#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
#define MALLOC_TARGET_IOS 1
#else // TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
#define MALLOC_TARGET_IOS 0
#endif // TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR

#ifdef __LP64__
#define MALLOC_TARGET_64BIT 1
#else // __LP64__
#define MALLOC_TARGET_64BIT 0
#endif

// <rdar://problem/12596555>
#if MALLOC_TARGET_IOS
# define CONFIG_MADVISE_PRESSURE_RELIEF 0
#else // MALLOC_TARGET_IOS
# define CONFIG_MADVISE_PRESSURE_RELIEF 1
#endif // MALLOC_TARGET_IOS

// <rdar://problem/12596555>
#define CONFIG_RECIRC_DEPOT 1
#define CONFIG_AGGRESSIVE_MADVISE 1

#if MALLOC_TARGET_IOS
# define DEFAULT_AGGRESSIVE_MADVISE_ENABLED true
#else // MALLOC_TARGET_IOS
# define DEFAULT_AGGRESSIVE_MADVISE_ENABLED false
#endif // MALLOC_TARGET_IOS

// <rdar://problem/10397726>
#define CONFIG_RELAXED_INVARIANT_CHECKS 1

// <rdar://problem/19818071>
#define CONFIG_MADVISE_STYLE MADV_FREE_REUSABLE

#if MALLOC_TARGET_64BIT
#define CONFIG_NANOZONE 1
#define CONFIG_ASLR_INTERNAL 0
#else // MALLOC_TARGET_64BIT
#define CONFIG_NANOZONE 0
#define CONFIG_ASLR_INTERNAL 1
#endif // MALLOC_TARGET_64BIT

// enable nano checking for corrupt free list
#define NANO_FREE_DEQUEUE_DILIGENCE 1

// This governs a last-free cache of 1 that bypasses the free-list for each region size
#define CONFIG_TINY_CACHE 1
#define CONFIG_SMALL_CACHE 1
#define CONFIG_MEDIUM_CACHE 1

// medium allocator enabled or disabled
#if MALLOC_TARGET_64BIT
#if MALLOC_TARGET_IOS
#define CONFIG_MEDIUM_ALLOCATOR 0
#else // MALLOC_TARGET_IOS
#define CONFIG_MEDIUM_ALLOCATOR 1
#endif // MALLOC_TARGET_IOS
#else // MALLOC_TARGET_64BIT
#define CONFIG_MEDIUM_ALLOCATOR 0
#endif // MALLOC_TARGET_64BIT

// The large last-free cache (aka. death row cache)
#if MALLOC_TARGET_IOS
#define CONFIG_LARGE_CACHE 0
#else
#define CONFIG_LARGE_CACHE 1
#endif

#if CONFIG_LARGE_CACHE
#define DEFAULT_LARGE_CACHE_ENABLED true
#endif

#if MALLOC_TARGET_IOS
// The VM system on iOS forces malloc-tagged memory to never be marked as
// copy-on-write, this would include calls we make to vm_copy. Given that the
// kernel would just be doing a memcpy, we force it to happen in userpsace.
#define CONFIG_REALLOC_CAN_USE_VMCOPY 0
#else
#define CONFIG_REALLOC_CAN_USE_VMCOPY 1
#endif

// memory resource exception handling
#if MALLOC_TARGET_IOS || TARGET_OS_SIMULATOR
#define ENABLE_MEMORY_RESOURCE_EXCEPTION_HANDLING 0
#else
#define ENABLE_MEMORY_RESOURCE_EXCEPTION_HANDLING 1
#endif

#if !TARGET_OS_DRIVERKIT && (!TARGET_OS_OSX || MALLOC_TARGET_64BIT)
#define CONFIG_FEATUREFLAGS_SIMPLE 1
#else
#define CONFIG_FEATUREFLAGS_SIMPLE 0
#endif

// presence of commpage memsize
#define CONFIG_HAS_COMMPAGE_MEMSIZE 1

// presence of commpage number of cpu count
#define CONFIG_HAS_COMMPAGE_NCPUS 1

// Use of hyper-shift for magazine selection.
#define CONFIG_NANO_USES_HYPER_SHIFT 0
#define CONFIG_TINY_USES_HYPER_SHIFT 0
#define CONFIG_SMALL_USES_HYPER_SHIFT 0

#endif // __PLATFORM_H
