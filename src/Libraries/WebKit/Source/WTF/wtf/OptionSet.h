/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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

#include <initializer_list>
#include <iterator>
#include <optional>
#include <type_traits>
#include <wtf/Assertions.h>
#include <wtf/EnumTraits.h>
#include <wtf/FastMalloc.h>
#include <wtf/MathExtras.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

template<typename E> class OptionSet;

// OptionSet is a class that represents a set of enumerators in a space-efficient manner. The enumerators
// must be powers of two greater than 0. This class is useful as a replacement for passing a bitmask of
// enumerators around.
template<typename E> class OptionSet {
    WTF_MAKE_FAST_ALLOCATED;
    static_assert(std::is_enum<E>::value, "T is not an enum type");

public:
    using StorageType = std::make_unsigned_t<std::underlying_type_t<E>>;

    template<typename StorageType> class Iterator {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        // Isolate the rightmost set bit.
        E operator*() const { return static_cast<E>(m_value & -m_value); }

        // Iterates from smallest to largest enum value by turning off the rightmost set bit.
        Iterator& operator++()
        {
            m_value &= m_value - 1;
            return *this;
        }

        Iterator& operator++(int) = delete;

        friend bool operator==(const Iterator&, const Iterator&) = default;

    private:
        Iterator(StorageType value) : m_value(value) { }
        friend OptionSet;

        StorageType m_value;
    };

    using iterator = Iterator<StorageType>;

    static constexpr OptionSet fromRaw(StorageType rawValue)
    {
        return OptionSet(static_cast<E>(rawValue), FromRawValue);
    }

    constexpr OptionSet() = default;

    constexpr OptionSet(E e)
        : m_storage(static_cast<StorageType>(e))
    {
        ASSERT(!m_storage || hasOneBitSet(static_cast<StorageType>(e)));
    }

    constexpr OptionSet(std::initializer_list<E> initializerList)
    {
        for (auto& option : initializerList) {
            ASSERT(hasOneBitSet(static_cast<StorageType>(option)));
            m_storage |= static_cast<StorageType>(option);
        }
    }

    constexpr OptionSet(std::optional<E> optional)
        : m_storage(optional ? static_cast<StorageType>(*optional) : 0)
    {
    }

    constexpr StorageType toRaw() const { return m_storage; }

    constexpr bool isEmpty() const { return !m_storage; }

    constexpr iterator begin() const { return m_storage; }
    constexpr iterator end() const { return 0; }

    constexpr explicit operator bool() const { return !isEmpty(); }

    constexpr bool contains(E option) const
    {
        return containsAny(option);
    }

    constexpr bool containsAny(OptionSet optionSet) const
    {
        return !!(*this & optionSet);
    }

    constexpr bool containsAll(OptionSet optionSet) const
    {
        return (*this & optionSet) == optionSet;
    }

    constexpr void add(OptionSet optionSet)
    {
        m_storage |= optionSet.m_storage;
    }

    constexpr void remove(OptionSet optionSet)
    {
        m_storage &= ~optionSet.m_storage;
    }

    constexpr void set(OptionSet optionSet, bool value)
    {
        if (value)
            add(optionSet);
        else
            remove(optionSet);
    }

    constexpr bool hasExactlyOneBitSet() const
    {
        return m_storage && !(m_storage & (m_storage - 1));
    }

    constexpr std::optional<E> toSingleValue() const
    {
        return hasExactlyOneBitSet() ? std::optional<E>(static_cast<E>(m_storage)) : std::nullopt;
    }

    friend constexpr bool operator==(const OptionSet&, const OptionSet&) = default;

    constexpr friend OptionSet operator|(OptionSet lhs, OptionSet rhs)
    {
        return fromRaw(lhs.m_storage | rhs.m_storage);
    }

    constexpr OptionSet& operator|=(const OptionSet& other)
    {
        add(other);
        return *this;
    }

    constexpr friend OptionSet operator&(OptionSet lhs, OptionSet rhs)
    {
        return fromRaw(lhs.m_storage & rhs.m_storage);
    }

    constexpr friend OptionSet operator-(OptionSet lhs, OptionSet rhs)
    {
        return fromRaw(lhs.m_storage & ~rhs.m_storage);
    }

    constexpr friend OptionSet operator^(OptionSet lhs, OptionSet rhs)
    {
        return fromRaw(lhs.m_storage ^ rhs.m_storage);
    }

    static OptionSet all() { return fromRaw(-1); }

private:
    enum InitializationTag { FromRawValue };
    constexpr OptionSet(E e, InitializationTag)
        : m_storage(static_cast<StorageType>(e))
    {
    }
    StorageType m_storage { 0 };
};

namespace IsValidOptionSetHelper {
template<typename T, typename E> struct OptionSetValueChecker;
template<typename T, typename E, E e, E... es>
struct OptionSetValueChecker<T, EnumValues<E, e, es...>> {
    static constexpr T allValidBits() { return static_cast<T>(e) | OptionSetValueChecker<T, EnumValues<E, es...>>::allValidBits(); }
};
template<typename T, typename E>
struct OptionSetValueChecker<T, EnumValues<E>> {
    static constexpr T allValidBits() { return 0; }
};
}

template<typename E>
WARN_UNUSED_RETURN constexpr bool isValidOptionSet(OptionSet<E> optionSet)
{
    // FIXME: Remove this when all OptionSet enums are migrated to generated serialization.
    auto allValidBitsValue = IsValidOptionSetHelper::OptionSetValueChecker<std::make_unsigned_t<std::underlying_type_t<E>>, typename EnumTraits<E>::values>::allValidBits();
    return (optionSet.toRaw() | allValidBitsValue) == allValidBitsValue;
}

} // namespace WTF

using WTF::OptionSet;
