/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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

#include <algorithm>
#include <wtf/StackPointer.h>
#include <wtf/ThreadingPrimitives.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

class StackBounds {
    WTF_MAKE_FAST_ALLOCATED;
public:

    // This 64k number was picked because a sampling of stack usage differences
    // between consecutive entries into one of the Interpreter::execute...()
    // functions was seen to be as high as 27k. Hence, 64k is chosen as a
    // conservative availability value that is not too large but comfortably
    // exceeds 27k with some buffer for error.
#if !ASAN_ENABLED
    static constexpr size_t DefaultReservedZone = 64 * 1024;
#else
    // ASAN inflates stack frames a lot. A factor of 3 was empirically found to
    // be the ratio of inflation of stack usage between 2 consecutive stack
    // recursion checkpoints. So, we'll also multiply the reserved zone size
    // accordingly to accommodate this.
    static constexpr size_t DefaultReservedZone = 64 * 1024 * 3;
#endif

    static constexpr StackBounds emptyBounds() { return StackBounds(); }

#if HAVE(STACK_BOUNDS_FOR_NEW_THREAD)
    // This function is only effective for newly created threads. In some platform, it returns a bogus value for the main thread.
    static StackBounds newThreadStackBounds(PlatformThreadHandle);
#endif
    static StackBounds currentThreadStackBounds()
    {
        auto result = currentThreadStackBoundsInternal();
        result.checkConsistency();
        return result;
    }

    void* origin() const
    {
        ASSERT(m_origin);
        return m_origin;
    }

    void* end() const
    {
        ASSERT(m_bound);
        return m_bound;
    }
    
    size_t size() const
    {
        return static_cast<char*>(m_origin) - static_cast<char*>(m_bound);
    }

    bool isEmpty() const { return !m_origin; }

    bool contains(void* p) const
    {
        if (isEmpty())
            return false;
        return (m_origin >= p) && (p > m_bound);
    }

    void* recursionLimit(size_t minReservedZone = DefaultReservedZone) const
    {
        checkConsistency();
        return static_cast<char*>(m_bound) + minReservedZone;
    }

    void* recursionLimit(char* startOfUserStack, size_t maxUserStack, size_t reservedZoneSize) const
    {
        checkConsistency();
        if (maxUserStack < reservedZoneSize)
            reservedZoneSize = maxUserStack;
        size_t maxUserStackWithReservedZone = maxUserStack - reservedZoneSize;

        char* endOfStackWithReservedZone = reinterpret_cast<char*>(m_bound) + reservedZoneSize;
        if (startOfUserStack < endOfStackWithReservedZone)
            return endOfStackWithReservedZone;
        size_t availableUserStack = startOfUserStack - endOfStackWithReservedZone;
        if (maxUserStackWithReservedZone > availableUserStack)
            maxUserStackWithReservedZone = availableUserStack;
        return startOfUserStack - maxUserStackWithReservedZone;
    }

    StackBounds withSoftOrigin(void* origin) const
    {
        ASSERT(contains(origin));
        return StackBounds(origin, m_bound);
    }

private:
    StackBounds(void* origin, void* end)
        : m_origin(origin)
        , m_bound(end)
    {
        ASSERT(isGrowingDownwards());
    }

    constexpr StackBounds()
        : m_origin(nullptr)
        , m_bound(nullptr)
    {
    }

    inline bool isGrowingDownwards() const
    {
        ASSERT(m_origin && m_bound);
        return m_bound <= m_origin;
    }

    WTF_EXPORT_PRIVATE static StackBounds currentThreadStackBoundsInternal();

    void checkConsistency() const
    {
#if ASSERT_ENABLED
        void* currentPosition = currentStackPointer();
        ASSERT(m_origin != m_bound);
        ASSERT(currentPosition < m_origin && currentPosition > m_bound);
#endif
    }

    void* m_origin;
    void* m_bound;

    friend class StackStats;
};

} // namespace WTF

using WTF::StackBounds;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
