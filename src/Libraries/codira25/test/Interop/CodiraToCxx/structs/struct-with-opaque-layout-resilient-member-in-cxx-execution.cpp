/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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

// RUN: %target-language-frontend %S/resilient-struct-in-cxx.code -enable-library-evolution -module-name Structs -emit-module -emit-module-path %t/Structs.codemodule

// RUN: %target-language-frontend %S/struct-with-opaque-layout-resilient-member-in-cxx.code -module-name UseStructs -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/useStructs.h -I %t


// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-structs-execution.o

// RUN: %target-interop-build-language -c %S/resilient-struct-in-cxx.code -enable-library-evolution -module-name Structs -o %t/resilient-struct-in-cxx.o -Xfrontend -entry-point-function-name -Xfrontend languageMain2

// RUN: %target-interop-build-language %S/struct-with-opaque-layout-resilient-member-in-cxx.code -o %t/language-structs-execution -Xlinker %t/resilient-struct-in-cxx.o -Xlinker %t/language-structs-execution.o -module-name UseStructs -Xfrontend -entry-point-function-name -Xfrontend languageMain -I %t

// RUN: %target-codesign %t/language-structs-execution
// RUN: %target-run %t/language-structs-execution | %FileCheck --check-prefixes=CHECK,CURRENT %s

// REQUIRES: executable_test

#include <assert.h>
#include "useStructs.h"

int main() {
  using namespace UseStructs;
  auto s = createUsesResilientSmallStruct();
  s.dump();
// CHECK: UsesResilientSmallStruct(97,FirstSmallStruct(x: 65)
  return 0;
}
