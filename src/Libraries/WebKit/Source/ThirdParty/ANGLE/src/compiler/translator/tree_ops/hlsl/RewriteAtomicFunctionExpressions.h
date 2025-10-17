/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteAtomicFunctionExpressions rewrites the expressions that contain
// atomic function calls and cannot be directly translated into HLSL into
// several simple ones that can be easily handled in the HLSL translator.
//
// We need to rewite these expressions because:
// 1. All GLSL atomic functions have return values, which all represent the
//    original value of the shared or ssbo variable; while all HLSL atomic
//    functions don't, and the original value can be stored in the last
//    parameter of the function call.
// 2. For HLSL atomic functions, the last parameter that stores the original
//    value is optional except for InterlockedExchange and
//    InterlockedCompareExchange. Missing original_value in the call of
//    InterlockedExchange or InterlockedCompareExchange results in a compile
//    error from HLSL compiler.
//
// RewriteAtomicFunctionExpressions is a function that can modify the AST
// to ensure all the expressions that contain atomic function calls can be
// directly translated into HLSL expressions.

#ifndef COMPILER_TRANSLATOR_TREEOPS_HLSL_REWRITE_ATOMIC_FUNCTION_EXPRESSIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_HLSL_REWRITE_ATOMIC_FUNCTION_EXPRESSIONS_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

[[nodiscard]] bool RewriteAtomicFunctionExpressions(TCompiler *compiler,
                                                    TIntermNode *root,
                                                    TSymbolTable *symbolTable,
                                                    int shaderVersion);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_HLSL_REWRITE_ATOMIC_FUNCTION_EXPRESSIONS_H_
