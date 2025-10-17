/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

// RUN: %target-language-frontend %S/language-class-virtual-method-dispatch.code -module-name Class -clang-header-expose-decls=all-public -typecheck -verify -emit-clang-header-path %t/class.h -enable-library-evolution

// RUN: %target-interop-build-clangxx -c %s -I %t -o %t/language-class-execution.o
// RUN: %target-interop-build-language %S/language-class-virtual-method-dispatch.code -o %t/language-class-execution -Xlinker %t/language-class-execution.o -module-name Class -Xfrontend -entry-point-function-name -Xfrontend languageMain -enable-library-evolution

// RUN: %target-codesign %t/language-class-execution
// RUN: %target-run %t/language-class-execution | %FileCheck %S/language-class-virtual-method-dispatch-execution.cpp

// REQUIRES: executable_test

#include "language-class-virtual-method-dispatch-execution.cpp"
