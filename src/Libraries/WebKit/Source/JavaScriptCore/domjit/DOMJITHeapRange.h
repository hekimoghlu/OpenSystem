/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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

#include <wtf/MathExtras.h>
#include <wtf/PrintStream.h>

namespace JSC { namespace DOMJIT {

class HeapRange {
public:
    constexpr HeapRange()
        : m_begin(UINT16_MAX)
        , m_end(UINT16_MAX)
    {
    }

    HeapRange(uint16_t begin, uint16_t end)
        : m_begin(begin)
        , m_end(end)
    {
        ASSERT_WITH_MESSAGE(begin <= end, "begin <= end is the invariant of this HeapRange.");
    }

    enum ConstExprTag { ConstExpr };
    constexpr HeapRange(ConstExprTag, uint16_t begin, uint16_t end)
        : m_begin(begin)
        , m_end(end)
    {
    }

    enum RawRepresentationTag { RawRepresentation };
    explicit constexpr HeapRange(RawRepresentationTag, uint32_t value)
        : m_raw(value)
    {
    }

    constexpr static HeapRange fromRaw(uint32_t value)
    {
        return HeapRange(RawRepresentation, value);
    }

    uint16_t begin() const { return m_begin; }
    uint16_t end() const { return m_end; }
    uint32_t rawRepresentation() { return m_raw; }

    constexpr explicit operator bool() const
    {
        return m_begin != m_end;
    }

    constexpr bool operator==(const HeapRange& other) const
    {
        return m_begin == other.m_begin && m_end == other.m_end;
    }

    template<uint16_t begin, uint16_t end>
    static constexpr HeapRange fromConstant()
    {
        static_assert(begin < end || (begin == UINT16_MAX && end == UINT16_MAX), "begin < end or the both are UINT16_MAX is the invariant of this HeapRange.");
        return HeapRange(ConstExpr, begin, end);
    }

    static constexpr HeapRange top() { return fromConstant<0, UINT16_MAX>(); }
    static constexpr HeapRange none() { return fromConstant<UINT16_MAX, UINT16_MAX>(); } // Empty range.

    bool isStrictSubtypeOf(const HeapRange& other) const
    {
        if (!*this || !other)
            return false;
        if (*this == other)
            return false;
        return other.m_begin <= m_begin && m_end <= other.m_end;
    }

    bool isSubtypeOf(const HeapRange& other) const
    {
        if (!*this || !other)
            return false;
        if (*this == other)
            return true;
        return isStrictSubtypeOf(other);
    }

    bool overlaps(const HeapRange& other) const
    {
        return WTF::rangesOverlap(m_begin, m_end, other.m_begin, other.m_end);
    }

    JS_EXPORT_PRIVATE void dump(PrintStream&) const;

private:
    union {
        struct {
            uint16_t m_begin;
            uint16_t m_end;
        };
        uint32_t m_raw;
    };
};

} }
