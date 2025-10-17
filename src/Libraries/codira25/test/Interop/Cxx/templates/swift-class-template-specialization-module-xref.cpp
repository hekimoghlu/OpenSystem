/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

// RUN: %empty-directory(%t)
// RUN: split-file %s %t
// RUN: %target-languagexx-frontend -emit-module %t/Inputs/test.code -module-name TestA -I %t/Inputs -o %t/test-part.codemodule
// RUN: %target-languagexx-frontend -merge-modules -emit-module %t/test-part.codemodule -module-name TestA -I %t/Inputs -o %t/TestA.codemodule -sil-verify-none
// RUN: %target-language-ide-test -print-module -module-to-print=TestA -I %t/ -source-filename=test -enable-experimental-cxx-interop | %FileCheck %s

//--- Inputs/module.modulemap
module CxxHeader {
    header "header.h"
    requires cplusplus
}

//--- Inputs/header.h

#include <stddef.h>

namespace std2 {

template<class T>
class vec {
public:
  using Element = T;
  using RawIterator = const T  * _Nonnull;
  vec() {}
  vec(const vec<T> &other) : items(other.items) { }
  ~vec() {}

  T * _Nonnull begin() {
    return items;
  }
  T * _Nonnull end() {
    return items + 10;
  }
  RawIterator begin() const {
    return items;
  }
  RawIterator end() const {
    return items + 10;
  }
  size_t size() const {
    return 10;
  }

private:
  T items[10];
};

} // namespace std2

namespace ns2 {

class App {
public:

  inline std2::vec<App> getApps() const {
    return {};
  }
  int x  = 0;
};

} // namespace ns2

using vec2Apps = std2::vec<ns2::App>;

//--- Inputs/test.code

import Cxx
import CxxHeader

extension vec2Apps : CxxSequence {
}

public fn testFunction() -> [Int] {
  let applications = ns2.App().getApps()
  return applications.map { Int($0.x) }
}

// CHECK: fn testFunction() -> [Int]
