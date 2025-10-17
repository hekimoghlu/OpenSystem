/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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

// Ensure: we don't hit any spurious warnings instantiating
// C++ standard library templates because of the generated header.

// RUN: %target-interop-build-clangxx -std=gnu++20 -fsyntax-only -c %t/test-stdlib.cpp -I %t -Wall -Werror -Werror=ignored-attributes -Wno-error=unused-command-line-argument

//--- print-string.code

public fn printString(_ s: String) {
}

//--- test-stdlib.cpp

#include "Stringer.h"
#include <iostream>

int main() {
  // Ensure that template instantiations inside C++
  // standard library in this call don't cause any new diagnostics
  // because of the generated header.
  std::cout << "test invoke std::cout\n" << std::endl;

  auto _ = language::String::init();
  return 0;
}
