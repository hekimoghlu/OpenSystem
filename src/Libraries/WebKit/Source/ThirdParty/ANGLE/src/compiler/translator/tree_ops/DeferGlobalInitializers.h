/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DeferGlobalInitializers is an AST traverser that moves global initializers into a separate
// function that is called in the beginning of main(). This enables initialization of globals with
// uniforms or non-constant globals, as allowed by the WebGL spec. Some initializers referencing
// non-constants may need to be unfolded into if statements in HLSL - this kind of steps should be
// done after DeferGlobalInitializers is run. Note that it's important that the function definition
// is at the end of the shader, as some globals may be declared after main().
//
// It can also initialize all uninitialized globals.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_DEFERGLOBALINITIALIZERS_H_
#define COMPILER_TRANSLATOR_TREEOPS_DEFERGLOBALINITIALIZERS_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TSymbolTable;

[[nodiscard]] bool DeferGlobalInitializers(TCompiler *compiler,
                                           TIntermBlock *root,
                                           bool initializeUninitializedGlobals,
                                           bool canUseLoopsToInitialize,
                                           bool highPrecisionSupported,
                                           bool forceDeferNonConstGlobalInitializers,
                                           TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_DEFERGLOBALINITIALIZERS_H_
