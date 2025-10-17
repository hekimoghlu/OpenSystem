/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

// RUN: %target-language-frontend %S/language-primitive-functions-cxx-bridging.code -module-name Functions -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/functions.h

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-functions-execution.o
// RUN: %target-interop-build-language %S/language-primitive-functions-cxx-bridging.code -o %t/language-functions-execution -Xlinker %t/language-functions-execution.o -module-name Functions -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-functions-execution
// RUN: %target-run %t/language-functions-execution

// REQUIRES: executable_test

#include <cassert>
#include "functions.h"

#define VERIFY_PASSTHROUGH_VALUE(function, value) assert(function(value) == (value));

int main() {
  using namespace Functions;

  VERIFY_PASSTHROUGH_VALUE(passThroughCBool, true);

  VERIFY_PASSTHROUGH_VALUE(passThroughCChar, 'a');
  VERIFY_PASSTHROUGH_VALUE(passThroughCWideChar, 'a');
  VERIFY_PASSTHROUGH_VALUE(passThroughCChar16, 0xFE1);
  VERIFY_PASSTHROUGH_VALUE(passThroughCChar32, 0x100FE);

  VERIFY_PASSTHROUGH_VALUE(passThroughCSignedChar, -1);
  VERIFY_PASSTHROUGH_VALUE(passThroughCShort, -512);
  VERIFY_PASSTHROUGH_VALUE(passThroughCInt, -999999);
  VERIFY_PASSTHROUGH_VALUE(passThroughCLongLong, -999998);

  VERIFY_PASSTHROUGH_VALUE(passThroughCUnsignedSignedChar, 255);
  VERIFY_PASSTHROUGH_VALUE(passThroughCUnsignedShort, 0xFFFF);
  VERIFY_PASSTHROUGH_VALUE(passThroughCUnsignedInt, 0xFFFFFFFF);
  VERIFY_PASSTHROUGH_VALUE(passThroughCUnsignedLongLong, 0xFFFFFFFF);

  VERIFY_PASSTHROUGH_VALUE(passThroughCFloat, 1.0f);
  VERIFY_PASSTHROUGH_VALUE(passThroughCDouble, 42.125f);

  VERIFY_PASSTHROUGH_VALUE(passThroughInt8, -1);
  VERIFY_PASSTHROUGH_VALUE(passThroughInt16, -512);
  VERIFY_PASSTHROUGH_VALUE(passThroughInt32, -999999);
  VERIFY_PASSTHROUGH_VALUE(passThroughInt64, -999999999999);

  VERIFY_PASSTHROUGH_VALUE(passThroughUInt8, 255);
  VERIFY_PASSTHROUGH_VALUE(passThroughUInt16, 0xffff);
  VERIFY_PASSTHROUGH_VALUE(passThroughUInt32, 0xffffffff);
  VERIFY_PASSTHROUGH_VALUE(passThroughUInt64, 0xffffffffffffffff);

  VERIFY_PASSTHROUGH_VALUE(passThroughFloat, 1.0f);
  VERIFY_PASSTHROUGH_VALUE(passThroughDouble, 42.125f);
  VERIFY_PASSTHROUGH_VALUE(passThroughFloat32, 1.0f);
  VERIFY_PASSTHROUGH_VALUE(passThroughFloat64, 42.125f);

  VERIFY_PASSTHROUGH_VALUE(passThroughInt, -999997);
  VERIFY_PASSTHROUGH_VALUE(passThroughUInt, 0xffffffff);
  VERIFY_PASSTHROUGH_VALUE(passThroughBool, true);
  VERIFY_PASSTHROUGH_VALUE(passThroughBool, false);

  int x = 0;
  VERIFY_PASSTHROUGH_VALUE(passThroughOpaquePointer, &x);
  VERIFY_PASSTHROUGH_VALUE(passThroughUnsafeRawPointer, &x);
  VERIFY_PASSTHROUGH_VALUE(passThroughUnsafeMutableRawPointer, &x);
  VERIFY_PASSTHROUGH_VALUE(roundTwoPassThroughUnsafeMutableRawPointer, nullptr);

  VERIFY_PASSTHROUGH_VALUE(passThroughUnsafeGenericMutableOptionalPointer, &x);
  VERIFY_PASSTHROUGH_VALUE(passThroughUnsafeGenericMutableOptionalPointer,
                           nullptr);
  VERIFY_PASSTHROUGH_VALUE(passThroughUnsafeGenericMutablePointer, &x);
  const int y = 0;
  VERIFY_PASSTHROUGH_VALUE(passThroughUnsafeGenericOptionalPointer, &x);
  VERIFY_PASSTHROUGH_VALUE(passThroughUnsafeGenericOptionalPointer, nullptr);
  VERIFY_PASSTHROUGH_VALUE(passThroughUnsafeGenericPointer, &x);
}
