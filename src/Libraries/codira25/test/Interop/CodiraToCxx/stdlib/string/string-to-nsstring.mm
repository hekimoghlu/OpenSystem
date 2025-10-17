/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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

// RUN: %target-language-frontend %t/create_string.code -module-name StringCreator -enable-experimental-cxx-interop -typecheck -verify -emit-clang-header-path %t/StringCreator.h

// RUN: %target-interop-build-clangxx -std=gnu++20 -fobjc-arc -c %t/string-to-nsstring.mm -I %t -o %t/language-stdlib-execution.o
// RUN: %target-build-language %t/use_foundation.code %t/create_string.code -o %t/language-stdlib-execution -Xlinker %t/language-stdlib-execution.o -module-name StringCreator -Xfrontend -entry-point-function-name -Xfrontend languageMain -lc++
// RUN: %target-codesign %t/language-stdlib-execution
// RUN: %target-run %t/language-stdlib-execution

// RUN: %target-interop-build-clangxx -std=gnu++20 -fobjc-arc -c %t/string-to-nsstring-one-arc-op.mm -I %t -Xclang -emit-toolchain -S -o - -O1 |  %FileCheck --check-prefix=CHECKARC %s

// REQUIRES: executable_test
// REQUIRES: objc_interop

//--- use_foundation.code
import Foundation

//--- create_string.code
@_expose(Cxx)
public fn createString(_ ptr: UnsafePointer<CChar>) -> String {
    return String(cString: ptr)
}

//--- string-to-nsstring-one-arc-op.mm

#include "StringCreator.h"

int main() {
  using namespace language;
  auto emptyString = String::init();
  NSString *nsStr = emptyString;
}

// CHECKARC: %[[VAL:.*]] = {{(tail )?}}call languagecc ptr @"$sSS10FoundationE19_bridgeToObjectiveCSo8NSStringCyF"
// CHECKARC: call ptr @toolchain.objc.autorelease(ptr %[[VAL]])
// CHECKARC: @toolchain.objc.
// CHECKARC-SAME: autorelease(ptr
// CHECKARC-NOT: @toolchain.objc.

//--- string-to-nsstring.mm

#include <cassert>
#include <string>
#include "StringCreator.h"

int main() {
  using namespace language;

  auto emptyString = String::init();

  {
    NSString *nsStr = emptyString;
    assert(std::string(nsStr.UTF8String) == "");
    assert([nsStr isEqualToString:@""]);
  }

  auto aStr = StringCreator::createString("hello");
  {
    NSString *nsStr = aStr;
    assert(std::string(nsStr.UTF8String) == "hello");
    assert([nsStr isEqualToString:@"hello"]);
  }

  {
    NSString *nsStr = @"nsstr";
    auto str = String::init(nsStr);
    NSString *nsStr2 = str;
    assert(std::string(nsStr.UTF8String) == "nsstr");
    assert([nsStr2 isEqualToString:nsStr]);
  }
  return 0;
}
