/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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

#include <optional>
#include <wtf/TriState.h>

namespace JSC {

class DefinePropertyAttributes {
public:
    static_assert(!static_cast<uint8_t>(TriState::False), "TriState::False is 0.");
    static_assert(static_cast<uint8_t>(TriState::True) == 1, "TriState::True is 1.");
    static_assert(static_cast<uint8_t>(TriState::Indeterminate) == 2, "TriState::Indeterminate is 2.");

    static constexpr unsigned ConfigurableShift = 0;
    static constexpr unsigned EnumerableShift = 2;
    static constexpr unsigned WritableShift = 4;
    static constexpr unsigned ValueShift = 6;
    static constexpr unsigned GetShift = 7;
    static constexpr unsigned SetShift = 8;

    DefinePropertyAttributes()
        : m_attributes(
            (static_cast<uint8_t>(TriState::Indeterminate) << ConfigurableShift)
            | (static_cast<uint8_t>(TriState::Indeterminate) << EnumerableShift)
            | (static_cast<uint8_t>(TriState::Indeterminate) << WritableShift)
            | (static_cast<uint8_t>(TriState::False) << ValueShift)
            | (static_cast<uint8_t>(TriState::False) << GetShift)
            | (static_cast<uint8_t>(TriState::False) << SetShift))
    {
    }

    explicit DefinePropertyAttributes(unsigned attributes)
        : m_attributes(attributes)
    {
    }

    unsigned rawRepresentation() const
    {
        return m_attributes;
    }

    bool hasValue() const
    {
        return m_attributes & (0b1 << ValueShift);
    }

    void setValue()
    {
        m_attributes = m_attributes | (0b1 << ValueShift);
    }

    bool hasGet() const
    {
        return m_attributes & (0b1 << GetShift);
    }

    void setGet()
    {
        m_attributes = m_attributes | (0b1 << GetShift);
    }

    bool hasSet() const
    {
        return m_attributes & (0b1 << SetShift);
    }

    void setSet()
    {
        m_attributes = m_attributes | (0b1 << SetShift);
    }

    bool hasWritable() const
    {
        return extractTriState(WritableShift) != TriState::Indeterminate;
    }

    std::optional<bool> writable() const
    {
        if (!hasWritable())
            return std::nullopt;
        return extractTriState(WritableShift) == TriState::True;
    }

    bool hasConfigurable() const
    {
        return extractTriState(ConfigurableShift) != TriState::Indeterminate;
    }

    std::optional<bool> configurable() const
    {
        if (!hasConfigurable())
            return std::nullopt;
        return extractTriState(ConfigurableShift) == TriState::True;
    }

    bool hasEnumerable() const
    {
        return extractTriState(EnumerableShift) != TriState::Indeterminate;
    }

    std::optional<bool> enumerable() const
    {
        if (!hasEnumerable())
            return std::nullopt;
        return extractTriState(EnumerableShift) == TriState::True;
    }

    void setWritable(bool value)
    {
        fillWithTriState(value ? TriState::True : TriState::False, WritableShift);
    }

    void setConfigurable(bool value)
    {
        fillWithTriState(value ? TriState::True : TriState::False, ConfigurableShift);
    }

    void setEnumerable(bool value)
    {
        fillWithTriState(value ? TriState::True : TriState::False, EnumerableShift);
    }

private:
    void fillWithTriState(TriState state, unsigned shift)
    {
        unsigned mask = 0b11 << shift;
        m_attributes = (m_attributes & ~mask) | (static_cast<uint8_t>(state) << shift);
    }

    TriState extractTriState(unsigned shift) const
    {
        return static_cast<TriState>((m_attributes >> shift) & 0b11);
    }

    unsigned m_attributes;
};


} // namespace JSC
