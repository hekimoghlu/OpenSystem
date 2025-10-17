/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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

// RUN: %target-language-frontend %S/nested-classes-in-cxx.code -enable-library-evolution -typecheck -module-name Classes -clang-header-expose-decls=all-public -emit-clang-header-path %t/classes.h

// RUN: %target-interop-build-clangxx -std=c++17 -c %s -I %t -o %t/language-classes-execution.o

// RUN: %target-interop-build-language %S/nested-classes-in-cxx.code -enable-library-evolution -o %t/language-classes-execution -Xlinker %t/language-classes-execution.o -module-name Classes -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-classes-execution
// RUN: %target-run %t/language-classes-execution

// REQUIRES: executable_test

#include "classes.h"
#include <cassert>

int main() {
  using namespace Classes;
  auto x = makeRecordConfig();
  RecordConfig::File::Gate y = x.getGate();
  assert(y.getProp() == 80);
  assert(y.computeValue() == 160);
  RecordConfig::AudioFormat z = x.getFile().getFormat();
  assert(z == RecordConfig::AudioFormat::ALAC);
  RecordConfig::File::Gate g = RecordConfig::File::Gate::init();
}
