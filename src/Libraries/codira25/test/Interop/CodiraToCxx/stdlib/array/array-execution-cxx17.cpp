/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
// RUN: split-file %S/array-execution.cpp %t

// RUN: %target-language-frontend %t/use-array.code -module-name UseArray -enable-experimental-cxx-interop -typecheck -verify -emit-clang-header-path %t/UseArray.h

// RUN: %target-interop-build-clangxx -fno-exceptions -std=gnu++17 -c %t/array-execution.cpp -I %t -o %t/language-stdlib-execution.o
// RUN: %target-build-language %t/use-array.code -o %t/language-stdlib-execution -Xlinker %t/language-stdlib-execution.o -module-name UseArray -Xfrontend -entry-point-function-name -Xfrontend languageMain
// RUN: %target-codesign %t/language-stdlib-execution
// RUN: %target-run %t/language-stdlib-execution | %FileCheck %S/array-execution.cpp

// REQUIRES: executable_test
