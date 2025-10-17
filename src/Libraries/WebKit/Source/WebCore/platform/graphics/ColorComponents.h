/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include <array>
#include <cmath>
#include <tuple>
#include <wtf/MathExtras.h>

namespace WebCore {

template<typename T, size_t N>
struct ColorComponents {
    constexpr static size_t Size = N;

    constexpr ColorComponents()
        : components { }
    {
    }

    template<typename ...Ts>
    constexpr ColorComponents(Ts ...input)
        : components {{ input ... }}
    {
        static_assert(sizeof...(Ts) == N);
    }

    template<typename F>
    constexpr auto map(F&& function) const -> ColorComponents<decltype(function(std::declval<T>())), N>;

    constexpr ColorComponents& operator+=(const ColorComponents&);
    constexpr ColorComponents& operator/=(T);

    constexpr ColorComponents operator+(T) const;
    constexpr ColorComponents operator/(T) const;
    constexpr ColorComponents operator*(T) const;

    constexpr ColorComponents abs() const;

    constexpr T& operator[](size_t i) { return components[i]; }
    constexpr const T& operator[](size_t i) const { return components[i]; }

    template<size_t I>
    constexpr T get() const;

    template<size_t Start, size_t End>
    constexpr ColorComponents<T, End - Start> subset() const;

    constexpr size_t size() const { return Size; }

    std::array<T, N> components;
};

template<typename T, typename ...Ts>
ColorComponents(T, Ts...) -> ColorComponents<T, 1 + sizeof...(Ts)>;

template<typename F, typename T, typename... Ts>
constexpr auto mapColorComponents(F&& function, T component, Ts... components) -> ColorComponents<decltype(function(component[0], components[0]...)), T::Size>
{
    static_assert(std::conjunction_v<std::bool_constant<Ts::Size == T::Size>...>, "All ColorComponents passed to mapColorComponents must have the same size");

    ColorComponents<decltype(function(component[0], components[0]...)), T::Size> result;
    for (std::remove_const_t<decltype(T::Size)> i = 0; i < T::Size; ++i)
        result[i] = function(component[i], components[i]...);
    return result;
}

template<typename T, size_t N>
template<typename F>
constexpr auto ColorComponents<T, N>::map(F&& function) const -> ColorComponents<decltype(function(std::declval<T>())), N>
{
    return mapColorComponents(std::forward<F>(function), *this);
}

template<typename T, size_t N>
constexpr ColorComponents<T, N>& ColorComponents<T, N>::operator+=(const ColorComponents& rhs)
{
    *this = mapColorComponents([](T c1, T c2) { return c1 + c2; }, *this, rhs);
    return *this;
}

template<typename T, size_t N>
constexpr ColorComponents<T, N>& ColorComponents<T, N>::operator/=(T rhs)
{
    *this = (*this / rhs);
    return *this;
}

template<typename T, size_t N>
constexpr ColorComponents<T, N> ColorComponents<T, N>::operator+(T rhs) const
{
    return map([rhs](T c) { return c + rhs; });
}

template<typename T, size_t N>
constexpr ColorComponents<T, N> ColorComponents<T, N>::operator/(T denominator) const
{
    return map([denominator](T c) { return c / denominator; });
}

template<typename T, size_t N>
constexpr ColorComponents<T, N> ColorComponents<T, N>::operator*(T factor) const
{
    return map([factor](T c) { return c * factor; });
}

template<typename T, size_t N>
constexpr ColorComponents<T, N> ColorComponents<T, N>::abs() const
{
    return map([](T c) { return std::abs(c); });
}

template<typename T, size_t N>
template<size_t I>
constexpr T ColorComponents<T, N>::get() const
{
    return components[I];
}

template<typename T, size_t N>
template<size_t Start, size_t End>
constexpr ColorComponents<T, End - Start> ColorComponents<T, N>::subset() const
{
    ColorComponents<T, End - Start> result;
    for (size_t i = Start; i < End; ++i)
        result[i - Start] = components[i];
    return result;
}

template<typename T, size_t N>
constexpr ColorComponents<T, N> perComponentMax(const ColorComponents<T, N>& a, const ColorComponents<T, N>& b)
{
    return mapColorComponents([](T c1, T c2) { return std::max(c1, c2); }, a, b);
}

template<typename T, size_t N>
constexpr ColorComponents<T, N> perComponentMin(const ColorComponents<T, N>& a, const ColorComponents<T, N>& b)
{
    return mapColorComponents([](T c1, T c2) { return std::min(c1, c2); }, a, b);
}

template<typename T, size_t N>
constexpr bool operator==(const ColorComponents<T, N>& a, const ColorComponents<T, N>& b)
{
    for (size_t i = 0; i < N; ++i) {
        if (a[i] == b[i])
            continue;
        if (isNaNConstExpr(a[i]) && isNaNConstExpr(b[i]))
            continue;
        return false;
    }

    return true;
}

} // namespace WebCore

namespace std {

template<typename T, size_t N>
class tuple_size<WebCore::ColorComponents<T, N>> : public std::integral_constant<size_t, N> {
};

template<size_t I, typename T, size_t N>
class tuple_element<I, WebCore::ColorComponents<T, N>> {
public:
    using type = T;
};

}
