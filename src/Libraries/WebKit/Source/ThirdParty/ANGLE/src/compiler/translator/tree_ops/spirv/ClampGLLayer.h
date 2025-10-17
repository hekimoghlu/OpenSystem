/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ClampGLLayer: Set gl_Layer to 0 if framebuffer is not layered.  The GL spec says:
//
// > A layer number written by a geometry shader has no effect if the framebuffer is not layered.
//
// While the Vulkan spec says:
//
// > If the Layer value is less than 0 or greater than or equal to the number of layers in the
// > framebuffer, then primitives may still be rasterized, fragment shaders may be executed, and the
// > framebuffer values for all layers are undefined.
//
// ANGLE sets gl_Layer to 0 if the framebuffer is not layered in this transformation.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_SPIRV_CLAMPGLLAYER_H_
#define COMPILER_TRANSLATOR_TREEOPS_SPIRV_CLAMPGLLAYER_H_

#include "common/angleutils.h"
#include "compiler/translator/Compiler.h"

namespace sh
{
class TCompiler;
class TIntermBlock;
class TSymbolTable;
class DriverUniform;

[[nodiscard]] bool ClampGLLayer(TCompiler *compiler,
                                TIntermBlock *root,
                                TSymbolTable *symbolTable,
                                const DriverUniform *driverUniforms);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SPIRV_CLAMPGLLAYER_H_
