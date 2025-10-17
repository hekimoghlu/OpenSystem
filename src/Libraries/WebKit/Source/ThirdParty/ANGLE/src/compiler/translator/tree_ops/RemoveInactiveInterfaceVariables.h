/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveInactiveInterfaceVariables.h:
//  Drop shader interface variable declarations for those that are inactive.  This step needs to be
//  done after CollectVariables.  This avoids having to emulate them (e.g. atomic counters for
//  Vulkan) or remove them in glslang wrapper (again, for Vulkan).
//
//  Shouldn't be run for the GL backend, as it queries shader interface variables from GL itself,
//  instead of relying on what was gathered during CollectVariables.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_REMOVEINACTIVEVARIABLES_H_
#define COMPILER_TRANSLATOR_TREEOPS_REMOVEINACTIVEVARIABLES_H_

#include "common/angleutils.h"

namespace sh
{

struct InterfaceBlock;
struct ShaderVariable;
class TCompiler;
class TIntermBlock;
class TSymbolTable;

[[nodiscard]] bool RemoveInactiveInterfaceVariables(
    TCompiler *compiler,
    TIntermBlock *root,
    TSymbolTable *symbolTable,
    const std::vector<sh::ShaderVariable> &attributes,
    const std::vector<sh::ShaderVariable> &inputVaryings,
    const std::vector<sh::ShaderVariable> &outputVariables,
    const std::vector<sh::ShaderVariable> &uniforms,
    const std::vector<sh::InterfaceBlock> &interfaceBlocks,
    bool removeFragmentOutputs);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REMOVEINACTIVEVARIABLES_H_
