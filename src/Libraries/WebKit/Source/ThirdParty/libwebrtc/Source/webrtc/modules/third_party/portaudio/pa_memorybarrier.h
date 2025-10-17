/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
 * The text above constitutes the entire PortAudio license; however,
 * the PortAudio community also makes the following non-binding requests:
 *
 * Any person wishing to distribute modifications to the Software is
 * requested to send the modifications to the original developer so that
 * they can be incorporated into the canonical version. It is also
 * requested that these non-binding requests be included along with the
 * license above.
 */

/**
 @file pa_memorybarrier.h
 @ingroup common_src
*/

/****************
 * Some memory barrier primitives based on the system.
 * right now only OS X, FreeBSD, and Linux are supported. In addition to
 *providing memory barriers, these functions should ensure that data cached in
 *registers is written out to cache where it can be snooped by other CPUs. (ie,
 *the volatile keyword should not be required)
 *
 * the primitives that must be defined are:
 *
 * PaUtil_FullMemoryBarrier()
 * PaUtil_ReadMemoryBarrier()
 * PaUtil_WriteMemoryBarrier()
 *
 ****************/

#ifndef MODULES_THIRD_PARTY_PORTAUDIO_PA_MEMORYBARRIER_H_
#define MODULES_THIRD_PARTY_PORTAUDIO_PA_MEMORYBARRIER_H_

#if defined(__APPLE__)
/* Support for the atomic library was added in C11.
 */
#if (__STDC_VERSION__ < 201112L) || defined(__STDC_NO_ATOMICS__)
#include <libkern/OSAtomic.h>
/* Here are the memory barrier functions. Mac OS X only provides
   full memory barriers, so the three types of barriers are the same,
   however, these barriers are superior to compiler-based ones.
   These were deprecated in MacOS 10.12. */
#define PaUtil_FullMemoryBarrier() OSMemoryBarrier()
#define PaUtil_ReadMemoryBarrier() OSMemoryBarrier()
#define PaUtil_WriteMemoryBarrier() OSMemoryBarrier()
#else
#include <stdatomic.h>
#define PaUtil_FullMemoryBarrier() atomic_thread_fence(memory_order_seq_cst)
#define PaUtil_ReadMemoryBarrier() atomic_thread_fence(memory_order_acquire)
#define PaUtil_WriteMemoryBarrier() atomic_thread_fence(memory_order_release)
#endif
#elif defined(__GNUC__)
/* GCC >= 4.1 has built-in intrinsics. We'll use those */
#if (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 1)
#define PaUtil_FullMemoryBarrier() __sync_synchronize()
#define PaUtil_ReadMemoryBarrier() __sync_synchronize()
#define PaUtil_WriteMemoryBarrier() __sync_synchronize()
/* as a fallback, GCC understands volatile asm and "memory" to mean it
 * should not reorder memory read/writes */
/* Note that it is not clear that any compiler actually defines __PPC__,
 * it can probably removed safely. */
#elif defined(__ppc__) || defined(__powerpc__) || defined(__PPC__)
#define PaUtil_FullMemoryBarrier() asm volatile("sync" ::: "memory")
#define PaUtil_ReadMemoryBarrier() asm volatile("sync" ::: "memory")
#define PaUtil_WriteMemoryBarrier() asm volatile("sync" ::: "memory")
#elif defined(__i386__) || defined(__i486__) || defined(__i586__) || \
    defined(__i686__) || defined(__x86_64__)
#define PaUtil_FullMemoryBarrier() asm volatile("mfence" ::: "memory")
#define PaUtil_ReadMemoryBarrier() asm volatile("lfence" ::: "memory")
#define PaUtil_WriteMemoryBarrier() asm volatile("sfence" ::: "memory")
#else
#ifdef ALLOW_SMP_DANGERS
#warning Memory barriers not defined on this system or system unknown
#warning For SMP safety, you should fix this.
#define PaUtil_FullMemoryBarrier()
#define PaUtil_ReadMemoryBarrier()
#define PaUtil_WriteMemoryBarrier()
#else
#           error Memory barriers are not defined on this system. You can still compile by defining ALLOW_SMP_DANGERS, but SMP safety will not be guaranteed.
#endif
#endif
#elif (_MSC_VER >= 1400) && !defined(_WIN32_WCE)
#include <intrin.h>
#pragma intrinsic(_ReadWriteBarrier)
#pragma intrinsic(_ReadBarrier)
#pragma intrinsic(_WriteBarrier)
/* note that MSVC intrinsics _ReadWriteBarrier(), _ReadBarrier(),
 * _WriteBarrier() are just compiler barriers *not* memory barriers */
#define PaUtil_FullMemoryBarrier() _ReadWriteBarrier()
#define PaUtil_ReadMemoryBarrier() _ReadBarrier()
#define PaUtil_WriteMemoryBarrier() _WriteBarrier()
#elif defined(_WIN32_WCE)
#define PaUtil_FullMemoryBarrier()
#define PaUtil_ReadMemoryBarrier()
#define PaUtil_WriteMemoryBarrier()
#elif defined(_MSC_VER) || defined(__BORLANDC__)
#define PaUtil_FullMemoryBarrier() _asm { lock add    [esp], 0}
#define PaUtil_ReadMemoryBarrier() _asm { lock add    [esp], 0}
#define PaUtil_WriteMemoryBarrier() _asm { lock add    [esp], 0}
#else
#ifdef ALLOW_SMP_DANGERS
#warning Memory barriers not defined on this system or system unknown
#warning For SMP safety, you should fix this.
#define PaUtil_FullMemoryBarrier()
#define PaUtil_ReadMemoryBarrier()
#define PaUtil_WriteMemoryBarrier()
#else
#       error Memory barriers are not defined on this system. You can still compile by defining ALLOW_SMP_DANGERS, but SMP safety will not be guaranteed.
#endif
#endif

#endif /* MODULES_THIRD_PARTY_PORTAUDIO_PA_MEMORYBARRIER_H_ */
