/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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

#include "Compiler.h"
#include "DataLog.h"
#include "HashMap.h"
#include "Lock.h"
#include "StackShot.h"
#include "StackTrace.h"

#include <atomic>

namespace WTF {

// This file contains some tools for tracking references.
// See Strong.h for an example of how to use it.

#if ENABLE(REFTRACKER)
template<typename T>
struct RefTrackerLoggingDisabledScope {
    WTF_MAKE_NONCOPYABLE(RefTrackerLoggingDisabledScope);
    RefTrackerLoggingDisabledScope()
    {
        ++T::refTrackerSingleton().loggingDisabledDepth;
    }
    ~RefTrackerLoggingDisabledScope()
    {
        --T::refTrackerSingleton().loggingDisabledDepth;
    }
};

struct RefTracker {
    WTF_MAKE_FAST_ALLOCATED;
public:
    RefTracker() = default;
    ~RefTracker() = default;

    // NEVER_INLINE to make skipping frames more predictable.
    WTF_EXPORT_PRIVATE void reportLive(void*);
    WTF_EXPORT_PRIVATE void reportDead(void*);
    WTF_EXPORT_PRIVATE void logAllLiveReferences();

    Lock lock { };
    UncheckedKeyHashMap<void*, std::unique_ptr<StackShot>> map WTF_GUARDED_BY_LOCK(lock) { };
    std::atomic<int> loggingDisabledDepth { };
};

template<typename T>
struct RefTrackerMixin final {
    ALWAYS_INLINE RefTrackerMixin()
    {
        RELEASE_ASSERT(!originalThis);
        originalThis = this;
        if (UNLIKELY(T::enabled()))
            T::refTrackerSingleton().reportLive(static_cast<void*>(this));
    }

    ALWAYS_INLINE RefTrackerMixin(RefTrackerMixin<T>&&)
    {
        RELEASE_ASSERT(!originalThis);
        originalThis = this;
        if (UNLIKELY(T::enabled()))
            T::refTrackerSingleton().reportLive(static_cast<void*>(this));
    }

    ALWAYS_INLINE RefTrackerMixin(const RefTrackerMixin<T>&)
    {
        RELEASE_ASSERT(!originalThis);
        originalThis = this;
        if (UNLIKELY(T::enabled()))
            T::refTrackerSingleton().reportLive(static_cast<void*>(this));
    }

    ALWAYS_INLINE ~RefTrackerMixin()
    {
        RELEASE_ASSERT(originalThis == this);
        if (UNLIKELY(T::enabled()))
            T::refTrackerSingleton().reportDead(static_cast<void*>(this));
    }

    RefTrackerMixin& operator=(const RefTrackerMixin& o)
    {
        RELEASE_ASSERT(o.originalThis == &o);
        RELEASE_ASSERT(originalThis == this);
        return *this;
    }

    RefTrackerMixin& operator=(RefTrackerMixin&& o)
    {
        RELEASE_ASSERT(o.originalThis == &o);
        RELEASE_ASSERT(originalThis == this);
        return *this;
    }

    // This guards against seeing an unconstructed object (say, if we are zero-initialized)
    RefTrackerMixin* originalThis = nullptr;
};

#define REFTRACKER_DECL(T, initializer) \
    struct T final { \
        inline static bool enabled() { \
            initializer \
            return Options::enable ## T(); \
        } \
        WTF_EXPORT_PRIVATE static WTF::RefTracker& refTrackerSingleton(); \
    };

#define REFTRACKER_MEMBERS(T) \
    WTF::RefTrackerMixin<T> m_refTrackerData;

#define REFTRACKER_IMPL(T) \
    WTF::RefTracker& T::refTrackerSingleton() \
    { \
        static WTF::LazyNeverDestroyed<WTF::RefTracker> s_singleton; \
        static std::once_flag s_onceFlag; \
        std::call_once(s_onceFlag, \
            [] { \
                s_singleton.construct(); \
            }); \
        return s_singleton.get(); \
    }

#else // ENABLE(REFTRACKER)

#define REFTRACKER_DECL(_, initializer)
#define REFTRACKER_MEMBERS(_)
#define REFTRACKER_IMPL(_)

#endif

} // namespace WTF
