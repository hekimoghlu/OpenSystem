/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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

#include "DisallowScope.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/ThreadSpecific.h>

namespace JSC {

class Heap;
class VM;

class DeferGC {
    WTF_MAKE_NONCOPYABLE(DeferGC);
    WTF_FORBID_HEAP_ALLOCATION;
public:
    DeferGC(VM&);
    ~DeferGC();

private:
    VM& m_vm;
};

class DeferGCForAWhile {
    WTF_MAKE_NONCOPYABLE(DeferGCForAWhile);
    WTF_FORBID_HEAP_ALLOCATION;
public:
    DeferGCForAWhile(VM&);
    ~DeferGCForAWhile();

private:
    JSC::Heap& m_heap;
};

class DisallowGC : public DisallowScope<DisallowGC> {
    WTF_MAKE_NONCOPYABLE(DisallowGC);
    WTF_FORBID_HEAP_ALLOCATION;
    typedef DisallowScope<DisallowGC> Base;
public:
#if ASSERT_ENABLED
    DisallowGC() = default;

    static void initialize()
    {
        s_scopeReentryCount.construct();
    }

private:
    static unsigned scopeReentryCount()
    {
        return *s_scopeReentryCount.get();
    }
    static void setScopeReentryCount(unsigned value)
    {
        *s_scopeReentryCount.get() = value;
    }
    
    JS_EXPORT_PRIVATE static LazyNeverDestroyed<ThreadSpecific<unsigned, WTF::CanBeGCThread::True>> s_scopeReentryCount;

#else
    ALWAYS_INLINE DisallowGC() { } // We need this to placate Clang due to unused warnings.
    ALWAYS_INLINE static void initialize() { }
#endif // ASSERT_ENABLED
    
    friend class DisallowScope<DisallowGC>;
};

} // namespace JSC
