/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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

// RUN: %target-language-frontend %t/print-string.code -module-name Stringer -enable-experimental-cxx-interop -typecheck -verify -emit-clang-header-path %t/Stringer.h

// RUN: %target-interop-build-clangxx -std=gnu++20 -c %t/string-conversions.cpp -I %t -o %t/language-stdlib-execution.o -DDEBUG=1
// RUN: %target-build-language %t/print-string.code -o %t/language-stdlib-execution -Xlinker %t/language-stdlib-execution.o -module-name Stringer -Xfrontend -entry-point-function-name -Xfrontend languageMain %target-cxx-lib
// RUN: %target-codesign %t/language-stdlib-execution
// RUN: %target-run %t/language-stdlib-execution | %FileCheck %s

// Ensure that this works in optimized mode:

// RUN: %target-interop-build-clangxx -std=gnu++20 -c %t/string-conversions.cpp -I %t -o %t/language-stdlib-execution-opt.o -O
// RUN: %target-build-language %t/print-string.code -o %t/language-stdlib-execution-opt -Xlinker %t/language-stdlib-execution-opt.o -module-name Stringer -Xfrontend -entry-point-function-name -Xfrontend languageMain %target-cxx-lib -O
// RUN: %target-codesign %t/language-stdlib-execution-opt
// RUN: %target-run %t/language-stdlib-execution-opt | %FileCheck %s

// REQUIRES: executable_test

//--- print-string.code

@_expose(Cxx)
public fn printString(_ s: String) {
    print("'''\(s)'''")
}

//--- string-conversions.cpp

#include <cassert>
#include "Stringer.h"

int main() {
  using namespace language;
  using namespace Stringer;

  {
    auto s = String("Hello world");
    assert(s.getCount() == 11);
    printString(s);
    s.append(String("!test"));
    printString(s);
    printString(s.lowercased());
    printString(s.uppercased());
  }
// CHECK: '''Hello world'''
// CHECK: '''Hello world!test'''
// CHECK: '''hello world!test'''
// CHECK: '''HELLO WORLD!TEST'''
  return 0;
}
