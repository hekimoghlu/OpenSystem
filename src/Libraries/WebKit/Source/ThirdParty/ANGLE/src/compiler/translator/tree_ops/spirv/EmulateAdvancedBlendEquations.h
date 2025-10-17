/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// EmulateAdvancedBlendEquations.h: Emulate advanced blend equations by implicitly reading back from
// the color attachment (as an input attachment) and apply the equation function based on a uniform.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_SPIRV_EMULATEADVANCEDBLENDEQUATIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_SPIRV_EMULATEADVANCEDBLENDEQUATIONS_H_

#include "common/angleutils.h"
#include "compiler/translator/Compiler.h"
#include "compiler/translator/tree_ops/spirv/EmulateFramebufferFetch.h"

namespace sh
{

class TIntermBlock;
class TSymbolTable;
class DriverUniform;
class AdvancedBlendEquations;

// Declares the necessary input attachment (if not already for framebuffer fetch), loads from it and
// implements the specified advanced blend equation functions.  A driver uniform is used to select
// which function to use at runtime.
[[nodiscard]] bool EmulateAdvancedBlendEquations(
    TCompiler *compiler,
    TIntermBlock *root,
    TSymbolTable *symbolTable,
    const AdvancedBlendEquations &advancedBlendEquations,
    const DriverUniform *driverUniforms,
    InputAttachmentMap *inputAttachmentMapOut);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SPIRV_EMULATEADVANCEDBLENDEQUATIONS_H_
