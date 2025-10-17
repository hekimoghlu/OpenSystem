/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

#include <wtf/CompactVariantOperations.h>

namespace WTF {

// A `CompactVariant` acts like a `std::variant` with the following differences:
// - All alternatives must be pointers, smart pointers, have size of 56 bits or fewer, or be specialized for `CompactVariantTraits`.
// - Can only contain 254 or fewer alternatives.
// - Has a more limited API, only offering `holds_alternative()` for type checking and `switchOn()` for value access.

template<CompactVariantAlternative... Ts> class CompactVariant {
    static_assert(sizeof...(Ts) < 255);
    using StdVariant = std::variant<Ts...>;
    using Index = uint8_t;
    using Storage = uint64_t;
    using Operations = CompactVariantOperations<Ts...>;
public:
    template<typename U> constexpr CompactVariant(U&& value)
        requires
            std::constructible_from<StdVariant, U>
         && (!std::same_as<std::remove_cvref_t<U>, CompactVariant>)
         && (!IsStdInPlaceTypeV<std::remove_cvref_t<U>>)
         && (!IsStdInPlaceIndexV<std::remove_cvref_t<U>>)
    {
        m_data = Operations::template encode<typename VariantBestMatch<StdVariant, U>::type>(std::forward<U>(value));
    }

    template<typename T, typename... Args> explicit constexpr CompactVariant(std::in_place_type_t<T>, Args&&... args)
        requires std::constructible_from<StdVariant, T>
    {
        m_data = Operations::template encodeFromArguments<T>(std::forward<Args>(args)...);
    }

    template<size_t I, typename... Args> explicit constexpr CompactVariant(std::in_place_index_t<I>, Args&&... args)
        requires std::constructible_from<StdVariant, std::variant_alternative_t<I, StdVariant>>
    {
        m_data = Operations::template encodeFromArguments<std::variant_alternative_t<I, StdVariant>>(std::forward<Args>(args)...);
    }

    CompactVariant(const CompactVariant& other)
    {
        Operations::copy(m_data, other.m_data);
    }

    CompactVariant& operator=(const CompactVariant& other)
    {
        if (m_data == other.m_data)
            return *this;

        Operations::destruct(m_data);
        Operations::copy(m_data, other.m_data);

        return *this;
    }

    CompactVariant(CompactVariant&& other)
    {
        Operations::move(m_data, other.m_data);

        // Set `other` to the "moved from" state.
        other.m_data = Operations::movedFromDataValue;
    }

    CompactVariant& operator=(CompactVariant&& other)
    {
        if (m_data == other.m_data)
            return *this;

        Operations::destruct(m_data);
        Operations::move(m_data, other.m_data);

        // Set `other` to the "moved from" state.
        other.m_data = Operations::movedFromDataValue;

        return *this;
    }

    template<typename U> CompactVariant& operator=(U&& value)
        requires
            std::constructible_from<StdVariant, U>
         && (!std::same_as<std::remove_cvref_t<U>, CompactVariant>)
         && (!IsStdInPlaceTypeV<std::remove_cvref_t<U>>)
         && (!IsStdInPlaceIndexV<std::remove_cvref_t<U>>)
    {
        Operations::destruct(m_data);
        m_data = Operations::template encode<typename VariantBestMatch<StdVariant, U>::type>(std::forward<U>(value));

        return *this;
    }

    ~CompactVariant()
    {
        Operations::destruct(m_data);
    }

    void swap(CompactVariant& other)
    {
        std::swap(m_data, other.m_data);
    }

    template<typename T, typename... Args> void emplace(Args&&... args)
    {
        Operations::destruct(m_data);
        m_data = Operations::template encodeFromArguments<T>(std::forward<Args>(args)...);
    }

    template<size_t I, typename... Args> void emplace(Args&&... args)
    {
        Operations::destruct(m_data);
        m_data = Operations::template encodeFromArguments<std::variant_alternative_t<I, StdVariant>>(std::forward<Args>(args)...);
    }

    constexpr Index index() const
    {
        return Operations::decodedIndex(m_data);
    }

    constexpr bool valueless_by_move() const
    {
        return m_data == Operations::movedFromDataValue;
    }

    template<typename T> constexpr bool holdsAlternative() const
    {
        static_assert(alternativeIndexV<T, StdVariant> <= std::variant_size_v<StdVariant>);
        return index() == alternativeIndexV<T, StdVariant>;
    }

    template<size_t I> constexpr bool holdsAlternative() const
    {
        static_assert(I <= std::variant_size_v<StdVariant>);
        return index() == I;
    }

    template<typename... F> decltype(auto) switchOn(F&&... f) const
    {
        return Operations::constPayloadForData(m_data, std::forward<F>(f)...);
    }

    bool operator==(const CompactVariant& other) const
    {
        if (index() != other.index())
            return false;

        return typeForIndex<StdVariant>(index(), [&]<typename T>() {
            return Operations::template equal<T>(m_data, other.m_data);
        });
    }

private:
    // FIXME: Use a smaller data type if values are small enough / empty.
    Storage m_data;
};

// Utility for making a CompactVariant directly from a parameter pack of types.
template<typename... Ts> using CompactVariantWrapper = CompactVariant<Ts...>;

} // namespace WTF

using WTF::CompactVariant;
using WTF::CompactVariantWrapper;
