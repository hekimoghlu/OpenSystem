/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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

// RUN: %target-language-frontend %S/language-functions.code -module-name Functions -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/functions.h  -cxx-interoperability-mode=upcoming-language

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-functions-execution.o -g
// RUN: %target-interop-build-language %S/language-functions.code -o %t/language-functions-execution -Xlinker %t/language-functions-execution.o -module-name Functions -Xfrontend -entry-point-function-name -Xfrontend languageMain -g

// RUN: %target-codesign %t/language-functions-execution
// RUN: %target-run %t/language-functions-execution 2>&1 | %FileCheck %s

// REQUIRES: executable_test

#include <cassert>
#include <stdio.h>
#include "functions.h"

int main() {
  static_assert(noexcept(Functions::passVoidReturnVoid()), "noexcept function");
  static_assert(noexcept(Functions::_impl::$s9Functions014passVoidReturnC0yyF()),
                "noexcept function");

  Functions::passVoidReturnVoid();
  Functions::passIntReturnVoid(-1);
  assert(Functions::passTwoIntReturnIntNoArgLabel(1, 2) == 42);
  assert(Functions::passTwoIntReturnInt(1, 2) == 3);
  assert(Functions::passTwoIntReturnIntNoArgLabelParamName(1, 4) == 5);
  Functions::passVoidReturnNever();
  return 42;
}

// CHECK: passVoidReturnVoid
// CHECK-NEXT: passIntReturnVoid -1
// CHECK-NEXT: passTwoIntReturnIntNoArgLabel
// CHECK-NEXT: passVoidReturnNever
