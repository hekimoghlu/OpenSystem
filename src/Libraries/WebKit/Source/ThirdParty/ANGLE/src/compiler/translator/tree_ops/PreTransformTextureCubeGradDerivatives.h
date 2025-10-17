/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
// Copyright 2023 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Multiple GPU vendors have issues with transforming explicit cubemap
// derivatives onto the appropriate face. The workarounds are vendor-specific.

#ifndef COMPILER_TRANSLATOR_TREEOPS_PRETRANSFORMTEXTURECUBEGRADDERIVATIVES_H_
#define COMPILER_TRANSLATOR_TREEOPS_PRETRANSFORMTEXTURECUBEGRADDERIVATIVES_H_

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

// GLSL specs say the following regarding cube
// map sampling with explicit derivatives:
//
//   For the cube version, the partial derivatives of
//   P are assumed to be in the coordinate system used
//   before texture coordinates are projected onto the
//   appropriate cube face.
//
// Apple silicon expects them partially pre-projected
// onto the target face and written to certain vector
// components depending on the major axis.
[[nodiscard]] bool PreTransformTextureCubeGradDerivatives(TCompiler *compiler,
                                                          TIntermBlock *root,
                                                          TSymbolTable *symbolTable,
                                                          int shaderVersion);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_PRETRANSFORMTEXTURECUBEGRADDERIVATIVES_H_
