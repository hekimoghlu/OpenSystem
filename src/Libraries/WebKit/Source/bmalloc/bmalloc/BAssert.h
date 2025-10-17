/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#pragma once

#include "BPlatform.h"
#include "Logging.h"

#if BUSE(OS_LOG)
#include <os/log.h>
#endif

#if defined(NDEBUG) && (BOS(DARWIN) || BPLATFORM(PLAYSTATION))

#if BASAN_ENABLED
#define BBreakpointTrap()  __builtin_trap()
#elif BCPU(X86_64) || BCPU(X86)
#define BBreakpointTrap()  __asm__ volatile ("int3")
#elif BCPU(ARM_THUMB2)
#define BBreakpointTrap()  __asm__ volatile ("bkpt #0")
#elif BCPU(ARM64)
#define BBreakpointTrap()  __asm__ volatile ("brk #0xc471")
#else
#error "Unsupported CPU".
#endif

// Crash with a SIGTRAP i.e EXC_BREAKPOINT.
// We are not using __builtin_trap because it is only guaranteed to abort, but not necessarily
// trigger a SIGTRAP. Instead, we use inline asm to ensure that we trigger the SIGTRAP.
#define BCRASH() do { \
        BBreakpointTrap(); \
        __builtin_unreachable(); \
    } while (false)

#else // not defined(NDEBUG) && (BOS(DARWIN) || BPLATFORM(PLAYSTATION))

#if BASAN_ENABLED
#define BCRASH() __builtin_trap()
#else

#if defined(__GNUC__) // GCC or Clang
#define BCRASH() do { \
    *(int*)0xbbadbeef = 0; \
    __builtin_trap(); \
} while (0)
#else
#define BCRASH() do { \
    *(int*)0xbbadbeef = 0; \
    ((void(*)())0)(); \
} while (0)
#endif // defined(__GNUC__)
#endif // BASAN_ENABLED

#endif // defined(NDEBUG) && (BOS(DARWIN) || BPLATFORM(PLAYSTATION))

#define BASSERT_IMPL(x) do { \
    if (!(x)) \
        BCRASH(); \
} while (0)

#define RELEASE_BASSERT(x) BASSERT_IMPL(x)
#define RELEASE_BASSERT_NOT_REACHED() BCRASH()

#if BUSE(OS_LOG)
#define BMALLOC_LOGGING_PREFIX "bmalloc: "
#define BLOG_ERROR(format, ...) os_log_error(OS_LOG_DEFAULT, BMALLOC_LOGGING_PREFIX format, __VA_ARGS__)
#else
#define BLOG_ERROR(format, ...) bmalloc::reportAssertionFailureWithMessage(__FILE__, __LINE__, __PRETTY_FUNCTION__, format, __VA_ARGS__)
#endif

#if defined(NDEBUG)
#define RELEASE_BASSERT_WITH_MESSAGE(x, format, ...) BASSERT_IMPL(x)
#else
#define RELEASE_BASSERT_WITH_MESSAGE(x, format, ...) do { \
    if (!(x)) { \
        BLOG_ERROR("ASSERTION FAILED: " #x " :: " format, ##__VA_ARGS__); \
        BCRASH(); \
    } \
} while (0)
#endif

#define BUNUSED(x) ((void)x)

// ===== Release build =====

#if defined(NDEBUG)

#define BASSERT(x)

#define IF_DEBUG(x)

#endif // defined(NDEBUG)


// ===== Debug build =====

#if !defined(NDEBUG)

#define BASSERT(x) BASSERT_IMPL(x)

#define IF_DEBUG(x) (x)

#endif // !defined(NDEBUG)
