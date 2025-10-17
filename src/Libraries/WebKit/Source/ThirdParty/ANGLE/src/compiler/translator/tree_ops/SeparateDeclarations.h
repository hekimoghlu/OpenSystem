/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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

//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_SEPARATEDECLARATIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_SEPARATEDECLARATIONS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermBlock;

// Transforms declarations so that in the end each declaration contains only one declarator.
// This is useful as an intermediate step when initialization needs to be separated from
// declaration, or when things need to be unfolded out of the initializer.
// Examples:
// Input:
//     int a[1] = int[1](1), b[1] = int[1](2);
// Output:
//     int a[1] = int[1](1);
//     int b[1] = int[1](2);
// Input:
//    struct S { vec3 d; } a, b;
// Output:
//    struct S { vec3 d; } a;
//    S b;
// Input:
//    struct { vec3 d; } a;
// Output (note: no change):
//    struct { vec3 d; } a;
// Input:
//    struct { vec3 d; } a, b;
// Output:
//    struct s1234 { vec3 d; } a;
//    s1234 b;
// Input:
//   struct Foo { int a; } foo();
// Output:
//   struct Foo { int a; };
//   Foo foo();
// Input with separateCompoundStructDeclarations=true:
//    struct S { vec3 d; } a;
// Output:
//    struct S { vec3 d; };
//    S a;
// Input with separateCompoundStructDeclarations=true:
//    struct S { vec3 d; } a, b;
// Output:
//    struct S { vec3 d; };
//    S a;
//    S b;
// Input with separateCompoundStructDeclarations=true:
//    struct { vec3 d; } a, b;
// Output:
//    struct s1234 { vec3 d; };
//    s1234 a;
//    s1234 b;
// Input with separateCompoundStructDeclarations=true:
//    struct { vec3 d; } a;
// Output (note: now, changes):
//    struct s1234 { vec3 d; };
//    s1234 a;

[[nodiscard]] bool SeparateDeclarations(TCompiler &compiler,
                                        TIntermBlock &root,
                                        bool separateCompoundStructDeclarations);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SEPARATEDECLARATIONS_H_
