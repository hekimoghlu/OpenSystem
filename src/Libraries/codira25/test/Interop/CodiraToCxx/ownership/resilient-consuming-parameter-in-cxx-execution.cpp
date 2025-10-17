/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

// RUN: %target-language-frontend %S/consuming-parameter-in-cxx.code -module-name Init -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/consuming.h -enable-library-evolution

// RUN: %target-interop-build-clangxx -c %S/consuming-parameter-in-cxx-execution.cpp -I %t -o %t/language-consume-execution.o
// RUN: %target-interop-build-language %S/consuming-parameter-in-cxx.code -o %t/language-consume-execution-evo -Xlinker %t/language-consume-execution.o -module-name Init -Xfrontend -entry-point-function-name -Xfrontend languageMain -enable-library-evolution

// RUN: %target-codesign %t/language-consume-execution-evo
// RUN: %target-run %t/language-consume-execution-evo | %FileCheck %S/consuming-parameter-in-cxx-execution.cpp

// REQUIRES: executable_test
