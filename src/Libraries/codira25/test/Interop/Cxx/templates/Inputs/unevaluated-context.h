/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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

#ifndef TEST_INTEROP_CXX_TEMPLATES_INPUTS_UNEVALUATED_CONTEXT_H
#define TEST_INTEROP_CXX_TEMPLATES_INPUTS_UNEVALUATED_CONTEXT_H

template <typename _Tp>
_Tp __declval(long);

template <typename _Tp>
struct __declval_protector {
  static const bool __stop = false;
};

template <typename _Tp>
auto declval() noexcept -> decltype(__declval<_Tp>(0)) {
  static_assert(__declval_protector<_Tp>::__stop,
                "declval() must not be used!");
  return __declval<_Tp>(0);
}

inline void stillCalled() {
    static int x = 0;
}

template <class T>
class Vec {
public:
  void push_back(const T &__x) {
    if (!noexcept(declval<T *>()))
      ;
    stillCalled();
  }
};

inline void initVector() {
  Vec<int> vv;
  vv.push_back(0);
}

template <class T>
class UseDeclVal {
public:
    UseDeclVal() {}

    auto declTypeRet() const noexcept -> decltype(declval<T>().method()) {
        return T().method();
    }

    inline int callMethod() const {
        int x = declTypeRet();
        return x;
    }
};

struct StructWithMethod {
    inline int method() {
        return 42;
    }
};

using UseDeclValStruct = UseDeclVal<StructWithMethod>;

#endif // TEST_INTEROP_CXX_TEMPLATES_INPUTS_UNEVALUATED_CONTEXT_H
