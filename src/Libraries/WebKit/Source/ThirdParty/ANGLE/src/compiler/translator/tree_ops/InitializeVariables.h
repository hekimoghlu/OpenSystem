/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

#ifndef COMPILER_TRANSLATOR_TREEOPS_INITIALIZEVARIABLES_H_
#define COMPILER_TRANSLATOR_TREEOPS_INITIALIZEVARIABLES_H_

#include <GLSLANG/ShaderLang.h>

#include "compiler/translator/ExtensionBehavior.h"
#include "compiler/translator/IntermNode.h"

namespace sh
{
class TCompiler;
class TSymbolTable;

typedef std::vector<const TVariable *> InitVariableList;

// For all of the functions below: If canUseLoopsToInitialize is set, for loops are used instead of
// a large number of initializers where it can make sense, such as for initializing large arrays.

// Populate a sequence of assignment operations to initialize "initializedSymbol". initializedSymbol
// may be an array, struct or any combination of these, as long as it contains only basic types.
void CreateInitCode(const TIntermTyped *initializedSymbol,
                    bool canUseLoopsToInitialize,
                    bool highPrecisionSupported,
                    TIntermSequence *initCode,
                    TSymbolTable *symbolTable);

// Initialize all uninitialized local variables, so that undefined behavior is avoided.
[[nodiscard]] bool InitializeUninitializedLocals(TCompiler *compiler,
                                                 TIntermBlock *root,
                                                 int shaderVersion,
                                                 bool canUseLoopsToInitialize,
                                                 bool highPrecisionSupported,
                                                 TSymbolTable *symbolTable);

// This function can initialize all the types that CreateInitCode is able to initialize. All
// variables must be globals which can be found in the symbol table. For now it is used for the
// following two scenarios:
//   1. Initializing gl_Position;
//   2. Initializing output variables referred to in the shader source.
// Note: The type of each lvalue in an initializer is retrieved from the symbol table. gl_FragData
// requires special handling because the number of indices which can be initialized is determined by
// enabled extensions.
[[nodiscard]] bool InitializeVariables(TCompiler *compiler,
                                       TIntermBlock *root,
                                       const InitVariableList &vars,
                                       TSymbolTable *symbolTable,
                                       int shaderVersion,
                                       const TExtensionBehavior &extensionBehavior,
                                       bool canUseLoopsToInitialize,
                                       bool highPrecisionSupported);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_INITIALIZEVARIABLES_H_
