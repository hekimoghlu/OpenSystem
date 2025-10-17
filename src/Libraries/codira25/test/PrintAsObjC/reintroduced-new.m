/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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

// REQUIRES: objc_interop

// RUN: %empty-directory(%t)

// FIXME: BEGIN -enable-source-import hackaround
// RUN:  %target-language-frontend(mock-sdk: -sdk %S/../../test/Inputs/clang-importer-sdk -I %t) -emit-module -o %t %S/../../test/Inputs/clang-importer-sdk/language-modules/ObjectiveC.code -disable-objc-attr-requires-foundation-module
// FIXME: END -enable-source-import hackaround

// RUN: %target-language-frontend(mock-sdk: -sdk %S/../../test/Inputs/clang-importer-sdk -I %t) -emit-module -o %t %S/Inputs/reintroduced-new.code -language-version 4 -disable-objc-attr-requires-foundation-module -module-name main
// RUN: %target-language-frontend(mock-sdk: -sdk %S/../../test/Inputs/clang-importer-sdk -I %t) -parse-as-library %t/main.codemodule -typecheck -verify -emit-objc-header-path %t/generated.h -disable-objc-attr-requires-foundation-module -language-version 4
// RUN: not %clang -fsyntax-only -x objective-c %s -include %t/generated.h -fobjc-arc -fmodules -Werror -F %clang-importer-sdk-path/frameworks -isysroot %S/../../test/Inputs/clang-importer-sdk 2>&1 | %FileCheck -check-prefix=CHECK -check-prefix=CHECK-4 %s

// RUN: %target-language-frontend(mock-sdk: -sdk %S/../../test/Inputs/clang-importer-sdk -I %t) -emit-module -o %t %S/Inputs/reintroduced-new.code -disable-objc-attr-requires-foundation-module -module-name main -language-version 5
// RUN: %target-language-frontend(mock-sdk: -sdk %S/../../test/Inputs/clang-importer-sdk -I %t) -parse-as-library %t/main.codemodule -typecheck -verify -emit-objc-header-path %t/generated.h -disable-objc-attr-requires-foundation-module -language-version 5
// RUN: not %clang -fsyntax-only -x objective-c %s -include %t/generated.h -fobjc-arc -fmodules -Werror -F %clang-importer-sdk-path/frameworks -isysroot %S/../../test/Inputs/clang-importer-sdk 2>&1 | %FileCheck  -check-prefix=CHECK -check-prefix=CHECK-5 %s

// CHECK-NOT: error:

void test() {
  // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'init' is unavailable
  (void)[[Base alloc] init];
   // CHECK-NOT: error:
  (void)[[Sub alloc] init];
  // CHECK-4: :[[@LINE+2]]:{{[0-9]+}}: error: 'new' is deprecated: -init is unavailable
  // CHECK-5: :[[@LINE+1]]:{{[0-9]+}}: error: 'new' is unavailable: -init is unavailable
  (void)[Base new];
   // CHECK-NOT: error:
  (void)[Sub new];
}

// CHECK-NOT: error:
