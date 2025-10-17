/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_ENABLE_IF_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_ENABLE_IF_H

template <bool B, class T = void>
struct enable_if {};

template <class T>
struct enable_if<true, T> {
  typedef T type;
};

template <class T>
struct is_bool {
  static const bool value = false;
};

template <>
struct is_bool<bool> {
  static const bool value = true;
};

struct HasMethodWithEnableIf {
  template <typename T>
  typename enable_if<is_bool<T>::value, bool>::type onlyEnabledForBool(T t) const {
    return !t;
  }
};

struct HasConstructorWithEnableIf {
  template<class T, class _ = typename enable_if<is_bool<T>::value, bool>::type>
  HasConstructorWithEnableIf(const T &);
};

struct HasConstructorWithEnableIfUsed {
  template<class T, class U = typename enable_if<is_bool<T>::value, bool>::type>
  HasConstructorWithEnableIfUsed(const T &, const U &);
};

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_ENABLE_IF_H
