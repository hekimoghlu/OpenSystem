/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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
// FlagSamplersForTexelFetch.h: finds all instances of texelFetch used with a static reference to a
// sampler uniform, and flag that uniform as having been used with texelFetch
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_SPIRV_FLAGSAMPLERSWITHTEXELFETCH_H_
#define COMPILER_TRANSLATOR_TREEOPS_SPIRV_FLAGSAMPLERSWITHTEXELFETCH_H_

#include "GLSLANG/ShaderVars.h"
#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermBlock;
class TSymbolTable;

// This flags all samplers which are statically accessed by a texelFetch invokation- that is, the
// sampler is used as a direct argument to the call to texelFetch. Dynamic accesses, or accesses
// with any amount of indirection, are not counted.
[[nodiscard]] bool FlagSamplersForTexelFetch(TCompiler *compiler,
                                             TIntermBlock *root,
                                             TSymbolTable *symbolTable,
                                             std::vector<sh::ShaderVariable> *uniforms);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SPIRV_FLAGSAMPLERSWITHTEXELFETCH_H_
