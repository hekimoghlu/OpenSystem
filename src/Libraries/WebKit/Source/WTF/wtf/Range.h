/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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

#include <limits.h>
#include <wtf/MathExtras.h>
#include <wtf/PrintStream.h>

namespace WTF {

// Note that the 'begin' is inclusive, while the 'end' is exclusive. These two ranges are non-
// overlapping:
//
//     rangeA = 0...8
//     rangeB = 8...16

template<typename PassedType>
class Range {
    WTF_MAKE_FAST_ALLOCATED;
public:
    typedef PassedType Type;
    
    Range()
        : m_begin(0)
        , m_end(0)
    {
    }

    explicit Range(Type value)
        : m_begin(value)
        , m_end(value + 1)
    {
        ASSERT(m_end >= m_begin);
    }

    Range(Type begin, Type end)
        : m_begin(begin)
        , m_end(end)
    {
        ASSERT(m_end >= m_begin);
        if (m_begin == m_end) {
            // Canonicalize empty ranges.
            m_begin = 0;
            m_end = 0;
        }
    }

    static Range top()
    {
        return Range(std::numeric_limits<Type>::min(), std::numeric_limits<Type>::max());
    }

    friend bool operator==(const Range&, const Range&) = default;

    explicit operator bool() const { return m_begin != m_end; }

    Range operator|(const Range& other) const
    {
        if (!*this)
            return other;
        if (!other)
            return *this;
        return Range(
            std::min(m_begin, other.m_begin),
            std::max(m_end, other.m_end));
    }
    
    Range& operator|=(const Range& other)
    {
        return *this = *this | other;
    }
    
    Type begin() const { return m_begin; }
    Type end() const { return m_end; }

    Type distance() const { return end() - begin(); }

    bool overlaps(const Range& other) const
    {
        return WTF::rangesOverlap(m_begin, m_end, other.m_begin, other.m_end);
    }

    bool contains(Type point) const
    {
        return m_begin <= point && point < m_end;
    }

    void dump(PrintStream& out) const
    {
        if (*this == Range()) {
            out.print("Bottom");
            return;
        }
        if (*this == top()) {
            out.print("Top");
            return;
        }
        if (m_begin + 1 == m_end) {
            out.print(m_begin);
            return;
        }
        out.print(m_begin, "...", m_end);
    }

private:
    Type m_begin;
    Type m_end;
};

} // namespace WTF

using WTF::Range;
