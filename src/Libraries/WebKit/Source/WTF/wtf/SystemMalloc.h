/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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

#if PLATFORM(MAC) || PLATFORM(IOS_FAMILY)
// Probabilistic Guard Malloc is not really enabled on older platforms but opt those to system malloc too for consistency.
#define HAVE_PROBABILISTIC_GUARD_MALLOC 1
#endif

namespace WTF {

struct SystemMalloc {
    static void* malloc(size_t size)
    {
        auto* result = ::malloc(size);
        if (!result)
            CRASH();
        return result;
    }

    static void* tryMalloc(size_t size)
    {
        return ::malloc(size);
    }

    static void* zeroedMalloc(size_t size)
    {
        auto* result = ::malloc(size);
        if (!result)
            CRASH();
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        memset(result, 0, size);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        return result;
    }

    static void* tryZeroedMalloc(size_t size)
    {
        auto* result = ::malloc(size);
        if (!result)
            return nullptr;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        memset(result, 0, size);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        return result;
    }

    static void* realloc(void* p, size_t size)
    {
        auto* result = ::realloc(p, size);
        if (!result)
            CRASH();
        return result;
    }

    static void* tryRealloc(void* p, size_t size)
    {
        return ::realloc(p, size);
    }

    static void free(void* p)
    {
        ::free(p);
    }

    static constexpr ALWAYS_INLINE size_t nextCapacity(size_t capacity)
    {
        return capacity + capacity / 4 + 1;
    }
};

#if HAVE(PROBABILISTIC_GUARD_MALLOC)
using ProbabilisticGuardMalloc = SystemMalloc;
#endif

}

using WTF::SystemMalloc;
#if HAVE(PROBABILISTIC_GUARD_MALLOC)
using WTF::ProbabilisticGuardMalloc;
#endif
