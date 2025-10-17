/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include <wtf/Assertions.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

// A mixin for creating the various kinds of variable offsets that our engine supports.
template<typename T>
class GenericOffset {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(GenericOffset);
public:
    static constexpr unsigned invalidOffset = UINT_MAX;
    
    GenericOffset()
        : m_offset(invalidOffset)
    {
    }
    
    explicit GenericOffset(unsigned offset)
        : m_offset(offset)
    {
    }
    
    bool operator!() const { return m_offset == invalidOffset; }
    
    unsigned offsetUnchecked() const
    {
        return m_offset;
    }
    
    unsigned offset() const
    {
        ASSERT(m_offset != invalidOffset);
        return m_offset;
    }
    
    friend bool operator==(const GenericOffset&, const GenericOffset&) = default;
    bool operator<(const GenericOffset& other) const
    {
        return m_offset < other.m_offset;
    }
    bool operator>(const GenericOffset& other) const
    {
        return m_offset > other.m_offset;
    }
    bool operator<=(const GenericOffset& other) const
    {
        return m_offset <= other.m_offset;
    }
    bool operator>=(const GenericOffset& other) const
    {
        return m_offset >= other.m_offset;
    }
    
    T operator+(int value) const
    {
        return T(offset() + value);
    }
    T operator-(int value) const
    {
        return T(offset() - value);
    }
    T& operator+=(int value)
    {
        return *static_cast<T*>(this) = *this + value;
    }
    T& operator-=(int value)
    {
        return *static_cast<T*>(this) = *this - value;
    }
    
private:
    unsigned m_offset;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename T>, GenericOffset<T>);

} // namespace JSC
