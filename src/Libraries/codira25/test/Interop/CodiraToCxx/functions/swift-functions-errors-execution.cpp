/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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

// RUN: %target-language-frontend %S/language-functions-errors.code -module-name Functions -enable-experimental-cxx-interop -clang-header-expose-decls=has-expose-attr-or-stdlib -enable-experimental-feature GenerateBindingsForThrowingFunctionsInCXX -typecheck -verify -emit-clang-header-path %t/functions.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-functions-errors-execution.o -DLANGUAGE_CXX_INTEROP_EXPERIMENTAL_LANGUAGE_ERROR
// RUN: %target-interop-build-language %S/language-functions-errors.code -o %t/language-functions-errors-execution -Xlinker %t/language-functions-errors-execution.o -module-name Functions -Xfrontend -entry-point-function-name -Xfrontend languageMain -enable-experimental-feature GenerateBindingsForThrowingFunctionsInCXX

// RUN: %target-codesign %t/language-functions-errors-execution
// RUN: %target-run %t/language-functions-errors-execution | %FileCheck %s

// REQUIRES: executable_test
// REQUIRES: language_feature_GenerateBindingsForThrowingFunctionsInCXX
// UNSUPPORTED: OS=windows-msvc

// rdar://102167469
// UNSUPPORTED: CPU=arm64e

#include <cassert>
#include <cstdio>
#include "functions.h"

int main() {
  static_assert(!noexcept(Functions::emptyThrowFunction()), "noexcept function");
  static_assert(!noexcept(Functions::throwFunction()), "noexcept function");
  static_assert(!noexcept(Functions::throwFunctionWithReturn()), "noexcept function");

  try {
    Functions::emptyThrowFunction();
  } catch (language::Error& e) {
    printf("Exception\n");
  }
  try {
    Functions::throwFunction();
  } catch (language::Error& e) {
      auto errorOpt = e.as<Functions::NaiveErrors>();
      assert(errorOpt.isSome());

      auto errorVal = errorOpt.get();
      assert(errorVal == Functions::NaiveErrors::throwError);
      errorVal.getMessage();
  }
  try {
    Functions::throwFunctionWithReturn();
  } catch (language::Error& e) {
     printf("Exception\n");
  }
  try {
    Functions::throwFunctionWithNeverReturn();
  } catch (language::Error& e) {
     printf("Exception\n");
  }
  try {
    Functions::testDestroyedError();
  } catch(const language::Error &e) { }

  return 0;
}

// CHECK: passEmptyThrowFunction
// CHECK-NEXT: passThrowFunction
// CHECK-NEXT: throwError
// CHECK-NEXT: passThrowFunctionWithReturn
// CHECK-NEXT: Exception
// CHECK-NEXT: passThrowFunctionWithNeverReturn
// CHECK-NEXT: Exception
// CHECK-NEXT: Test destroyed
