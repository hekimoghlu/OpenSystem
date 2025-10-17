/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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
// SimplifyLoopConditions is an AST traverser that converts loop conditions and loop expressions
// to regular statements inside the loop. This way further transformations that generate statements
// from loop conditions and loop expressions work correctly.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_SIMPLIFYLOOPCONDITIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_SIMPLIFYLOOPCONDITIONS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

[[nodiscard]] bool SimplifyLoopConditions(TCompiler *compiler,
                                          TIntermNode *root,
                                          TSymbolTable *symbolTable);

[[nodiscard]] bool SimplifyLoopConditions(TCompiler *compiler,
                                          TIntermNode *root,
                                          unsigned int conditionsToSimplify,
                                          TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_SIMPLIFYLOOPCONDITIONS_H_
