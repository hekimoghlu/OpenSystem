/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

#ifndef TEST_INTEROP_CXX_CLASS_STRUCTURED_BINDINGS_H
#define TEST_INTEROP_CXX_CLASS_STRUCTURED_BINDINGS_H

namespace std {

template<class X, class Y>
struct pear {
    pear(X x, Y y) {}

    X getX() const {
        return 0;
    }
    Y getY() const {
        return 42;
    }
};

template<class T>
struct tuple_size {
    constexpr static const int value = 2;
};

template<int n, class T>
struct tuple_element {
};

template<class X, class Y>
struct tuple_element<0, pear<X, Y>> {
    using type = X;
};

template<class X, class Y>
struct tuple_element<1, pear<X, Y>> {
    using type = Y;
};

template<int N, class X, class Y>
inline typename tuple_element<N, pear<X, Y>>::type get(const pear<X, Y> & value) {
    if constexpr (N == 0) {
        return value.getX();
    } else {
        return value.getY();
    }
}

} // namespace std

inline int testDestructure(int x) {
    auto val = std::pear<int, int>( x, x + 2 );
    auto [y,z] = val;
    return z;
}

#endif // TEST_INTEROP_CXX_CLASS_STRUCTURED_BINDINGS_H
