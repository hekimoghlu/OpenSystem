/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 10, 2025.
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

#include <stdint.h>
#include <wtf/Platform.h>

enum class MemoryRestriction {
    kRwxToRw,
    kRwxToRx,
    kRwToRw,
    kRwToRo,
};

#if USE(APPLE_INTERNAL_SDK)
#include <WebKitAdditions/JSGlobalObjectAdditions.h>
#endif

#if defined(OS_THREAD_SELF_RESTRICT) != defined(OS_THREAD_SELF_RESTRICT_SUPPORTED)
#error Must override both or neither of OS_THREAD_SELF_RESTRICT and OS_THREAD_SELF_RESTRICT_SUPPORTED
#endif

#if defined(OS_THREAD_SELF_RESTRICT) && defined(OS_THREAD_SELF_RESTRICT_SUPPORTED)
template <MemoryRestriction restriction>
static ALWAYS_INLINE bool threadSelfRestrictSupported()
{
    return OS_THREAD_SELF_RESTRICT_SUPPORTED(restriction);
}

template <MemoryRestriction restriction>
static ALWAYS_INLINE void threadSelfRestrict()
{
    OS_THREAD_SELF_RESTRICT(restriction);
}

#else // Not defined(OS_THREAD_SELF_RESTRICT) && defined(OS_THREAD_SELF_RESTRICT_SUPPORTED)
#if OS(DARWIN) && CPU(ARM64)

#include "JSCConfig.h"

#include <wtf/Platform.h>

#if USE(INLINE_JIT_PERMISSIONS_API)
#include <BrowserEngineCore/BEMemory.h>

template <MemoryRestriction restriction>
static ALWAYS_INLINE bool threadSelfRestrictSupported()
{
    if constexpr ((restriction == MemoryRestriction::kRwxToRw)
        || (restriction == MemoryRestriction::kRwxToRx)) {
        return (&be_memory_inline_jit_restrict_with_witness_supported != nullptr
            && !!be_memory_inline_jit_restrict_with_witness_supported());
    }
    return false;
}

template <MemoryRestriction restriction>
static ALWAYS_INLINE void threadSelfRestrict()
{
    ASSERT(g_jscConfig.useFastJITPermissions);
    if constexpr (restriction == MemoryRestriction::kRwxToRw)
        be_memory_inline_jit_restrict_rwx_to_rw_with_witness();
    else if constexpr (restriction == MemoryRestriction::kRwxToRx)
        be_memory_inline_jit_restrict_rwx_to_rx_with_witness();
    else
        RELEASE_ASSERT_NOT_REACHED();
}

#elif USE(PTHREAD_JIT_PERMISSIONS_API)
#include <pthread.h>

template <MemoryRestriction restriction>
static ALWAYS_INLINE bool threadSelfRestrictSupported()
{
    if constexpr ((restriction == MemoryRestriction::kRwxToRw)
        || (restriction == MemoryRestriction::kRwxToRx)) {
        return !!pthread_jit_write_protect_supported_np();
    }
    return false;
}

template <MemoryRestriction restriction>
static ALWAYS_INLINE void threadSelfRestrict()
{
    ASSERT(g_jscConfig.useFastJITPermissions);
    if constexpr (restriction == MemoryRestriction::kRwxToRw)
        pthread_jit_write_protect_np(false);
    else if constexpr (restriction == MemoryRestriction::kRwxToRx)
        pthread_jit_write_protect_np(true);
    else
        RELEASE_ASSERT_NOT_REACHED();
}

#elif USE(APPLE_INTERNAL_SDK)
#include <os/thread_self_restrict.h>

template <MemoryRestriction restriction>
SUPPRESS_ASAN static ALWAYS_INLINE bool threadSelfRestrictSupported()
{
    if constexpr ((restriction == MemoryRestriction::kRwxToRw)
        || (restriction == MemoryRestriction::kRwxToRx)) {
        return !!os_thread_self_restrict_rwx_is_supported();
    }
    return false;
}

template <MemoryRestriction restriction>
static ALWAYS_INLINE void threadSelfRestrict()
{
    ASSERT(g_jscConfig.useFastJITPermissions);
    if constexpr (restriction == MemoryRestriction::kRwxToRw)
        os_thread_self_restrict_rwx_to_rw();
    else if constexpr (restriction == MemoryRestriction::kRwxToRx)
        os_thread_self_restrict_rwx_to_rx();
    else
        RELEASE_ASSERT_NOT_REACHED();
}

#else
template <MemoryRestriction>
static ALWAYS_INLINE bool threadSelfRestrictSupported()
{
    return false;
}

template <MemoryRestriction>
static ALWAYS_INLINE void threadSelfRestrict()
{
    bool tautologyToIgnoreWarning = true;
    if (tautologyToIgnoreWarning)
        RELEASE_ASSERT_NOT_REACHED();
}

#endif
#else // Not OS(DARWIN) && CPU(ARM64)
#if defined(OS_THREAD_SELF_RESTRICT) || defined(OS_THREAD_SELF_RESTRICT_SUPPORTED)
#error OS_THREAD_SELF_RESTRICT and OS_THREAD_SELF_RESTRICT_SUPPORTED are only used on ARM64+Darwin-only systems
#endif

template <MemoryRestriction>
static ALWAYS_INLINE bool threadSelfRestrictSupported()
{
    return false;
}

template <MemoryRestriction>
NO_RETURN_DUE_TO_CRASH ALWAYS_INLINE void threadSelfRestrict()
{
    CRASH();
}

#endif // OS(DARWIN) && CPU(ARM64)
#endif // defined(OS_THREAD_SELF_RESTRICT) && defined(OS_THREAD_SELF_RESTRICT_SUPPORTED)
