/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_VARIADIC_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_VARIADIC_H

template <class... Ts> struct Tuple {};

template <>
struct Tuple<> {
  void set() {}
};

template <class T, class... Ts>
struct Tuple<T, Ts...> : Tuple<Ts...> {
  Tuple(T t, Ts... ts) : Tuple<Ts...>(ts...), _t(t) {}

  void set(T t, Ts... ts) { _t = t; Tuple<Ts...>::set(ts...); }

  T first() { return _t; }
  Tuple<Ts...> rest() { return *this; }

  T _t;
};

struct IntWrapper {
  int value;
  int getValue() const { return value; }
};

typedef Tuple<IntWrapper> Single;
typedef Tuple<IntWrapper, IntWrapper> Pair;
typedef Tuple<IntWrapper, IntWrapper, IntWrapper> Triple;
typedef Tuple<Tuple<IntWrapper, IntWrapper>, Tuple<IntWrapper, IntWrapper>> Nested;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_VARIADIC_H
