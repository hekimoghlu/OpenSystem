/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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

#include <wtf/HashSet.h>
#include <wtf/Hasher.h>
#include <wtf/RetainPtr.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSRunLoop;
#endif

namespace WTF {

class SchedulePair : public ThreadSafeRefCounted<SchedulePair> {
public:
    static Ref<SchedulePair> create(CFRunLoopRef runLoop, CFStringRef mode) { return adoptRef(*new SchedulePair(runLoop, mode)); }

#if PLATFORM(COCOA)
    static Ref<SchedulePair> create(NSRunLoop* runLoop, CFStringRef mode) { return adoptRef(*new SchedulePair(runLoop, mode)); }
    NSRunLoop* nsRunLoop() const { return m_nsRunLoop.get(); }
#endif

    CFRunLoopRef runLoop() const { return m_runLoop.get(); }
    CFStringRef mode() const { return m_mode.get(); }

    WTF_EXPORT_PRIVATE bool operator==(const SchedulePair& other) const;

private:
    SchedulePair(CFRunLoopRef runLoop, CFStringRef mode)
        : m_runLoop(runLoop)
    {
        if (mode)
            m_mode = adoptCF(CFStringCreateCopy(nullptr, mode));
    }

#if PLATFORM(COCOA)
    WTF_EXPORT_PRIVATE SchedulePair(NSRunLoop*, CFStringRef);
    RetainPtr<NSRunLoop> m_nsRunLoop;
#endif

    RetainPtr<CFRunLoopRef> m_runLoop;
    RetainPtr<CFStringRef> m_mode;
};

inline void add(Hasher& hasher, const SchedulePair& pair)
{
    // FIXME: Hashing a CFHash here is unfortunate.
    add(hasher, pair.runLoop(), pair.mode() ? CFHash(pair.mode()) : 0);
}

struct SchedulePairHash {
    static unsigned hash(const SchedulePair* pair)
    {
        return computeHash(*pair);
    }

    static unsigned hash(const RefPtr<SchedulePair>& pair)
    {
        return computeHash(*pair);
    }

    static bool equal(const SchedulePair* a, const RefPtr<SchedulePair>& b) { return a == b; }
    static bool equal(const RefPtr<SchedulePair>& a, const RefPtr<SchedulePair>& b) { return a == b; }
    static bool equal(const RefPtr<SchedulePair>& a, const SchedulePair* b) { return a == b; }

    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

typedef UncheckedKeyHashSet<RefPtr<SchedulePair>, SchedulePairHash> SchedulePairHashSet;

} // namespace WTF

using WTF::SchedulePair;
using WTF::SchedulePairHashSet;
