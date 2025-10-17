/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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

#include <variant>
#include <wtf/StdLibExtras.h>
#include <wtf/VectorTraits.h>

namespace WTF {

// MARK: - Utility concepts/traits for std::variant.

template<typename>       struct VariantAllAlternativesCanCopyWithMemcpyHelper;
template<typename... Ts> struct VariantAllAlternativesCanCopyWithMemcpyHelper<std::variant<Ts...>> : std::integral_constant<bool, all<VectorTraits<Ts>::canCopyWithMemcpy...>> { };
template<typename V>     concept VariantAllAlternativesCanCopyWithMemcpy = VariantAllAlternativesCanCopyWithMemcpyHelper<V>::value;

template<typename>       struct VariantAllAlternativesCanMoveWithMemcpyHelper;
template<typename... Ts> struct VariantAllAlternativesCanMoveWithMemcpyHelper<std::variant<Ts...>> : std::integral_constant<bool, all<VectorTraits<Ts>::canMoveWithMemcpy...>> { };
template<typename V>     concept VariantAllAlternativesCanMoveWithMemcpy = VariantAllAlternativesCanMoveWithMemcpyHelper<V>::value;

// MARK: - Best match for std::variant construction.

// `VariantBestMatch` picks the type `T` from `Ts...` in `std::variant<Ts...>` that will be used when the
// `std::variant<Ts...>` is constructed from type `Arg`. Implementation based off of libc++.

struct VariantNoNarrowingCheck {
    template<typename D, typename S> using apply = std::type_identity_t<D>;
};
struct VariantNarrowingCheck {
    template<typename D> static auto test(D(&&)[1]) -> std::type_identity_t<D>;
    template<typename D, typename S> using apply = decltype(test<D>({ std::declval<S>() }));
};
template<typename D, typename S> using VariantCheckForNarrowing = typename std::conditional_t<std::is_arithmetic_v<D>, VariantNarrowingCheck, VariantNoNarrowingCheck>::template apply<D, S>;
template<typename T, size_t I> struct VariantOverload {
    template<typename U> auto operator()(T, U&&) const -> VariantCheckForNarrowing<T, U>;
};
template<typename... Bases> struct VariantAllOverloads : Bases... {
    void operator()() const;
    using Bases::operator()...;
};
template<typename Seq>   struct VariantMakeOverloadsImpl;
template<size_t... I> struct VariantMakeOverloadsImpl<std::index_sequence<I...> > {
    template<typename... Ts> using apply = VariantAllOverloads<VariantOverload<Ts, I>...>;
};
template<typename... Ts> using VariantMakeOverloads = typename VariantMakeOverloadsImpl<std::make_index_sequence<sizeof...(Ts)> >::template apply<Ts...>;
template<typename T, typename... Ts> using VariantBestMatchImpl = typename std::invoke_result_t<VariantMakeOverloads<Ts...>, T, T>;

template<typename V, typename Arg>     struct VariantBestMatch;
template<typename Arg, typename... Ts> struct VariantBestMatch<std::variant<Ts...>, Arg> {
    using type = VariantBestMatchImpl<Arg, Ts...>;
};

// MARK: - Type switching for std::variant

// Calls a zero argument functor with a template argument corresponding to the index's mapped type.
//
// e.g.
//   using Variant = std::variant<int, float>;
//
//   Variant foo = 5;
//   typeForIndex<Variant>(foo.index(), /* index will be 0 for first parameter, <int> */
//       []<typename T>() {
//           if constexpr (std::is_same_v<T, int>) {
//               print("we got an int");  <--- this will get called
//           } else if constexpr (std::is_same_v<T, float>) {
//               print("we got an float");  <--- this will NOT get called
//           }
//       }
//   );

template<typename V, size_t I = 0, typename F> constexpr ALWAYS_INLINE decltype(auto) visitTypeForIndex(size_t index, NOESCAPE F&& f)
{
    constexpr auto size = std::variant_size_v<V>;

    // To implement dispatch for variants, this recursively looping switch will work for
    // variants with any number of alternatives, with variants with greater than 32 recursing
    // and starting the switch at 32 (and so on).

#define WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT 32
#define WTF_VARIANT_EXTRAS_VISIT_CASE(N, D) \
        I + N:                                                                                      \
        {                                                                                           \
            if constexpr (I + N < size) {                                                           \
                return f.template operator()<std::variant_alternative_t<I + N, V>>();               \
            } else {                                                                                \
                WTF_UNREACHABLE();                                                                  \
            }                                                                                       \
        }                                                                                           \

    switch (index) {
    case WTF_VARIANT_EXTRAS_VISIT_CASE(0, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(1, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(2, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(3, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(4, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(5, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(6, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(7, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(8, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(9, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(10, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(11, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(12, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(13, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(14, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(15, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(16, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(17, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(18, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(19, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(20, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(21, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(22, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(23, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(24, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(25, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(26, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(27, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(28, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(29, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(30, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    case WTF_VARIANT_EXTRAS_VISIT_CASE(31, WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT)
    }

    constexpr auto nextI = std::min(I + WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT, size);

    if constexpr (nextI < size)
        return visitTypeForIndex<V, nextI>(index, std::forward<F>(f));

    WTF_UNREACHABLE();

#undef WTF_VARIANT_EXTRAS_VISIT_CASE_COUNT
#undef WTF_VARIANT_EXTRAS_VISIT_CASE
}

template<typename V, typename... F> constexpr auto typeForIndex(size_t index, F&&... f) -> decltype(visitTypeForIndex<V>(index, makeVisitor(std::forward<F>(f)...)))
{
    return visitTypeForIndex<V>(index, makeVisitor(std::forward<F>(f)...));
}

} // namespace WTF
