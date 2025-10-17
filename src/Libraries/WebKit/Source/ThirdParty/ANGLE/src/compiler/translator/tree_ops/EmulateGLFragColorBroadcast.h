/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
// Emulate gl_FragColor broadcast behaviors in ES2 where
// GL_EXT_draw_buffers is explicitly enabled in a fragment shader.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_EMULATEGLFRAGCOLORBROADCAST_H_
#define COMPILER_TRANSLATOR_TREEOPS_EMULATEGLFRAGCOLORBROADCAST_H_

#include <vector>

#include "common/angleutils.h"

namespace sh
{
struct ShaderVariable;
class TCompiler;
class TIntermBlock;
class TSymbolTable;

// Replace all gl_FragColor with gl_FragData[0], and in the end of main() function,
// assign gl_FragData[1] ... gl_FragData[maxDrawBuffers - 1] with gl_FragData[0].
// If gl_FragColor is in outputVariables, it is replaced by gl_FragData.
// Similarly replace all gl_SecondaryFragColorEXT with gl_SecondaryFragDataEXT[0].
[[nodiscard]] bool EmulateGLFragColorBroadcast(TCompiler *compiler,
                                               TIntermBlock *root,
                                               int maxDrawBuffers,
                                               int maxDualSourceDrawBuffers,
                                               std::vector<ShaderVariable> *outputVariables,
                                               TSymbolTable *symbolTable,
                                               int shaderVersion);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_EMULATEGLFRAGCOLORBROADCAST_H_
