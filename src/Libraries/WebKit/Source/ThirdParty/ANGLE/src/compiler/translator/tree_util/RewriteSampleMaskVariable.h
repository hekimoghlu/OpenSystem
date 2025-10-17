/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
// RewriteSampleMaskVariable.cpp: Find any references to gl_SampleMask and gl_SampleMaskIn, and
// rewrite it with ANGLESampleMask or ANGLESampleMaskIn.
//

#ifndef COMPILER_TRANSLATOR_TREEUTIL_REWRITESAMPLEMASKVARIABLE_H_
#define COMPILER_TRANSLATOR_TREEUTIL_REWRITESAMPLEMASKVARIABLE_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TSymbolTable;
class TIntermTyped;

// Rewrite every gl_SampleMask or gl_SampleMaskIn to "ANGLESampleMask" or "ANGLESampleMaskIn", then
// at the end of shader re-assign the values of this global variable to gl_SampleMask and
// gl_SampleMaskIn. This to solve the problem which the non constant index is used for the unsized
// array problem.
[[nodiscard]] bool RewriteSampleMask(TCompiler *compiler,
                                     TIntermBlock *root,
                                     TSymbolTable *symbolTable,
                                     const TIntermTyped *numSamplesUniform);

[[nodiscard]] bool RewriteSampleMaskIn(TCompiler *compiler,
                                       TIntermBlock *root,
                                       TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEUTIL_REWRITESAMPLEMASKVARIABLE_H_
