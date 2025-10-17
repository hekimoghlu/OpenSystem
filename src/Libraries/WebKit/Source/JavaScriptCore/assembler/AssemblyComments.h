/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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

#include "Options.h"

#include <optional>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/StdMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class AssemblyCommentRegistry {
    WTF_MAKE_TZONE_ALLOCATED(AssemblyCommentRegistry);
    WTF_MAKE_NONCOPYABLE(AssemblyCommentRegistry);
public:
    static AssemblyCommentRegistry& singleton();
    static void initialize();

    Lock& getLock() WTF_RETURNS_LOCK(m_lock) { return m_lock; }

    using CommentMap = UncheckedKeyHashMap<uintptr_t, String>;

    void registerCodeRange(void* start, void* end, CommentMap&& map)
    {
        if (LIKELY(!Options::needDisassemblySupport()) || !map.size())
            return;
        Locker locker { m_lock };

        auto newStart = std::bit_cast<uintptr_t>(start);
        auto newEnd = std::bit_cast<uintptr_t>(end);
        RELEASE_ASSERT(newStart < newEnd);

#if ASSERT_ENABLED
        for (auto it : m_comments) {
            auto thisStart = orderedKeyInverse(it.first);
            auto& [thisEnd, _] = it.second;
            ASSERT(newEnd <= thisStart
                || thisEnd <= newStart);
            ASSERT(thisStart < thisEnd);
        }
#else
        (void) newStart;
#endif

        m_comments.emplace(orderedKey(start), std::pair { newEnd, WTFMove(map) });
    }

    void unregisterCodeRange(void* start, void* end)
    {
        if (LIKELY(!Options::needDisassemblySupport()))
            return;
        Locker locker { m_lock };

        auto it = m_comments.find(orderedKey(start));
        if (it == m_comments.end())
            return;

        auto& [foundEnd, _] = it->second; 
        RELEASE_ASSERT(foundEnd == std::bit_cast<uintptr_t>(end));
        m_comments.erase(it);
    }

    inline std::optional<String> comment(void* in)
    {
        if (LIKELY(!Options::needDisassemblySupport()))
            return { };
        Locker locker { m_lock };
        auto it = m_comments.lower_bound(orderedKey(in));

        if (it == m_comments.end())
            return { };
        
        auto& [end, map] = it->second;
        if (std::bit_cast<uintptr_t>(in) > std::bit_cast<uintptr_t>(end))
            return { };

        auto it2 = map.find(std::bit_cast<uintptr_t>(in));

        if (it2 == map.end())
            return { };

        return { it2->value.isolatedCopy() };
    }

    AssemblyCommentRegistry() = default;

private:

    // Flip ordering for lower_bound comparator to work.
    inline uintptr_t orderedKey(void* in) { return std::numeric_limits<uintptr_t>::max() - std::bit_cast<uintptr_t>(in); }
    inline uintptr_t orderedKeyInverse(uintptr_t in) { return std::numeric_limits<uintptr_t>::max() - in; }

    Lock m_lock;
    StdMap<uintptr_t, std::pair<uintptr_t, CommentMap>> m_comments WTF_GUARDED_BY_LOCK(m_lock);
};

} // namespace JSC
