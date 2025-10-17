/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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

// RUN: %target-language-frontend %S/generic-struct-in-cxx.code -enable-library-evolution -module-name Generics -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/generics.h

// RUN: %target-interop-build-clangxx -std=gnu++20 -c %s -I %t -o %t/language-generics-execution.o
// RUN: %target-interop-build-language %S/generic-struct-in-cxx.code -o %t/language-generics-execution -Xlinker %t/language-generics-execution.o -enable-library-evolution -module-name Generics -Xfrontend -entry-point-function-name -Xfrontend languageMain

// RUN: %target-codesign %t/language-generics-execution
// RUN: %target-run %t/language-generics-execution | %FileCheck %S/generic-struct-execution.cpp

// REQUIRES: executable_test

#include "generic-struct-execution.cpp"
