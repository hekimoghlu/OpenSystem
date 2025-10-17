/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_EAGER_INSTANTIATION_PROBLEMS_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_EAGER_INSTANTIATION_PROBLEMS_H

struct MagicNumber {
  int getInt() const { return 42; }
};

template<class T>
struct MagicWrapper {
  void callGetInt() const {
    T::getIntDoesNotExist();
  }

  template <typename A> int sfinaeGetInt(A a, decltype(&A::getInt)) {
    return a.getInt();
  }
  template <typename A> int sfinaeGetInt(A a, ...) {
    return -42;
  }
};

typedef MagicWrapper<int> BrokenMagicWrapper;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_CLASS_TEMPLATE_EAGER_INSTANTIATION_PROBLEMS_H
