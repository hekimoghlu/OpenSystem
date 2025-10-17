/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ReplaceClipCullDistanceVariable.h: Find any references to gl_ClipDistance or gl_CullDistance and
// replace it with ANGLEClipDistance or ANGLECullDistance.
//

#ifndef COMPILER_TRANSLATOR_TREEUTIL_REPLACECLIPCULLDISTANCEVARIABLE_H_
#define COMPILER_TRANSLATOR_TREEUTIL_REPLACECLIPCULLDISTANCEVARIABLE_H_

#include "GLSLANG/ShaderLang.h"
#include "common/angleutils.h"

namespace sh
{

struct InterfaceBlock;
class TCompiler;
class TIntermBlock;
class TSymbolTable;
class TIntermTyped;

// Replace every gl_ClipDistance assignment with assignment to "ANGLEClipDistance",
// then at the end of shader re-assign the values of this global variable to gl_ClipDistance.
// This to solve some complex usages such as user passing gl_ClipDistance as output reference
// to a function.
// Furthermore, at the end shader, some disabled gl_ClipDistance[i] can be skipped from the
// assignment.
[[nodiscard]] bool ReplaceClipDistanceAssignments(TCompiler *compiler,
                                                  TIntermBlock *root,
                                                  TSymbolTable *symbolTable,
                                                  const GLenum shaderType,
                                                  const TIntermTyped *clipDistanceEnableFlags);

[[nodiscard]] bool ReplaceCullDistanceAssignments(TCompiler *compiler,
                                                  TIntermBlock *root,
                                                  TSymbolTable *symbolTable,
                                                  const GLenum shaderType);

[[nodiscard]] bool ZeroDisabledClipDistanceAssignments(TCompiler *compiler,
                                                       TIntermBlock *root,
                                                       TSymbolTable *symbolTable,
                                                       const GLenum shaderType,
                                                       const TIntermTyped *clipDistanceEnableFlags);
}  // namespace sh

#endif
