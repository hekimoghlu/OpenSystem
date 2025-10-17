/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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

#include <wtf/ForbidHeapAllocation.h>
#include <wtf/Noncopyable.h>

namespace JSC {

template<class T>
class DisallowScope {
    WTF_MAKE_NONCOPYABLE(DisallowScope);
    WTF_FORBID_HEAP_ALLOCATION;
public:
#if ASSERT_ENABLED
    DisallowScope()
    {
        auto count = T::scopeReentryCount();
        T::setScopeReentryCount(++count);
    }

    ~DisallowScope()
    {
        auto count = T::scopeReentryCount();
        ASSERT(count);
        T::setScopeReentryCount(--count);
    }

    static bool isInEffectOnCurrentThread()
    {
        return !!T::scopeReentryCount();
    }

#else // not ASSERT_ENABLED
    ALWAYS_INLINE DisallowScope() { } // We need this to placate Clang due to unused warnings.
    ALWAYS_INLINE static bool isInEffectOnCurrentThread() { return false; }
#endif // ASSERT_ENABLED
};

} // namespace JSC
