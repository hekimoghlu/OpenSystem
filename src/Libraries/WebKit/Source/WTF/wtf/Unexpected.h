/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
// Implementation of Library Fundamentals v3's std::expected, as described here: http://wg21.link/p0323r4

#pragma once

/*
    unexpected synopsis

namespace std {
namespace experimental {
inline namespace fundamentals_v3 {
    // ?.?.3, Unexpected object type
    template <class E>
      class unexpected;

    // ?.?.4, Unexpected relational operators
    template <class E>
        constexpr bool
        operator==(const unexpected<E>&, const unexpected<E>&);
    template <class E>
        constexpr bool
        operator!=(const unexpected<E>&, const unexpected<E>&);

    template <class E>
    class unexpected {
    public:
        unexpected() = delete;
        template <class U = E>
          constexpr explicit unexpected(E&&);
        constexpr const E& value() const &;
        constexpr E& value() &;
        constexpr E&& value() &&;
        constexpr E const&& value() const&&;
    private:
        E val; // exposition only
    };

}}}

*/

#include <cstdlib>
#include <utility>
#include <wtf/StdLibExtras.h>

namespace std {
namespace experimental {
inline namespace fundamentals_v3 {

template<class E>
class unexpected {
    WTF_MAKE_FAST_ALLOCATED;
public:
    unexpected() = delete;
    template <class U = E>
    constexpr explicit unexpected(U&& u) : val(std::forward<U>(u)) { }
    constexpr const E& value() const & { return val; }
    constexpr E& value() & { return val; }
    constexpr E&& value() && { return WTFMove(val); }
    constexpr const E&& value() const && { return WTFMove(val); }

private:
    E val;
};

template<class E> constexpr bool operator==(const unexpected<E>& lhs, const unexpected<E>& rhs) { return lhs.value() == rhs.value(); }

}}} // namespace std::experimental::fundamentals_v3

template<class E> using Unexpected = std::experimental::unexpected<E>;

// Not in the std::expected spec, but useful to work around lack of C++17 deduction guides.
template<class E> constexpr Unexpected<std::decay_t<E>> makeUnexpected(E&& v) { return Unexpected<typename std::decay<E>::type>(std::forward<E>(v)); }
