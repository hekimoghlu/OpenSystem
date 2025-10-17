/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
// Scalarize vector and matrix constructor args, so that vectors built from components don't have
// matrix arguments, and matrices built from components don't have vector arguments. This avoids
// driver bugs around vector and matrix constructors.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_GLSL_SCALARIZEVECANDMATCONSTRUCTORARGS_H_
#define COMPILER_TRANSLATOR_TREEOPS_GLSL_SCALARIZEVECANDMATCONSTRUCTORARGS_H_

#include "GLSLANG/ShaderLang.h"
#include "common/angleutils.h"
#include "common/debug.h"

namespace sh
{
class TCompiler;
class TIntermBlock;
class TSymbolTable;

#ifdef ANGLE_ENABLE_GLSL
[[nodiscard]] bool ScalarizeVecAndMatConstructorArgs(TCompiler *compiler,
                                                     TIntermBlock *root,
                                                     TSymbolTable *symbolTable);
#else
[[nodiscard]] ANGLE_INLINE bool ScalarizeVecAndMatConstructorArgs(TCompiler *compiler,
                                                                  TIntermBlock *root,
                                                                  TSymbolTable *symbolTable)
{
    UNREACHABLE();
    return false;
}
#endif

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_GLSL_SCALARIZEVECANDMATCONSTRUCTORARGS_H_
