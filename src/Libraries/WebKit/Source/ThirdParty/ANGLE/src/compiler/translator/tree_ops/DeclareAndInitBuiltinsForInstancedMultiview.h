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

//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Regardless of the shader type, the following AST transformations are applied:
// - Add declaration of View_ID_OVR.
// - Replace every occurrence of gl_ViewID_OVR with ViewID_OVR, mark ViewID_OVR as internal and
// declare it with the same qualifier.
//
// If the shader type is a vertex shader, the following AST transformations are applied:
// - Replace every occurrence of gl_InstanceID with InstanceID, mark InstanceID as internal and set
// its qualifier to EvqTemporary.
// - Add initializers of ViewID_OVR and InstanceID to the beginning of the body of main. The pass
// should be executed before any variables get collected so that usage of gl_InstanceID is recorded.
// - If the output is ESSL or GLSL and the selectViewInNvGLSLVertexShader option is
// enabled, the expression
//   gl_Layer = int(ViewID_OVR) + multiviewBaseViewLayerIndex;
// is added after ViewID and InstanceID are initialized. Also, multiviewBaseViewLayerIndex is added
// as a uniform.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_DECLAREANDINITBUILTINSFORINSTANCEDMULTIVIEW_H_
#define COMPILER_TRANSLATOR_TREEOPS_DECLAREANDINITBUILTINSFORINSTANCEDMULTIVIEW_H_

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TSymbolTable;

[[nodiscard]] bool DeclareAndInitBuiltinsForInstancedMultiview(
    TCompiler *compiler,
    TIntermBlock *root,
    unsigned numberOfViews,
    GLenum shaderType,
    const ShCompileOptions &compileOptions,
    ShShaderOutput shaderOutput,
    TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_DECLAREANDINITBUILTINSFORINSTANCEDMULTIVIEW_H_
