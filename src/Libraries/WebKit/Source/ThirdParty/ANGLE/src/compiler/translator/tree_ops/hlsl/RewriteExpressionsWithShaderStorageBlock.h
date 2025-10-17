/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
// RewriteExpressionsWithShaderStorageBlock rewrites the expressions that contain shader storage
// block calls into several simple ones that can be easily handled in the HLSL translator. After the
// AST pass, all ssbo related blocks will be like below:
//     ssbo_access_chain = ssbo_access_chain;
//     ssbo_access_chain = expr_no_ssbo;
//     lvalue_no_ssbo    = ssbo_access_chain;
//
// Below situations are needed to be rewritten (Details can be found in .cpp file).
//     SSBO as the operand of compound assignment operators.
//     SSBO as the operand of ++/--.
//     SSBO as the operand of repeated assignment.
//     SSBO as the operand of readonly unary/binary/ternary operators.
//     SSBO as the argument of aggregate type.
//     SSBO as the condition of if/switch/while/do-while/for

#ifndef COMPILER_TRANSLATOR_TREEOPS_HLSL_REWRITE_EXPRESSIONS_WITH_SHADER_STORAGE_BLOCK_H_
#define COMPILER_TRANSLATOR_TREEOPS_HLSL_REWRITE_EXPRESSIONS_WITH_SHADER_STORAGE_BLOCK_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;
class TSymbolTable;

[[nodiscard]] bool RewriteExpressionsWithShaderStorageBlock(TCompiler *compiler,
                                                            TIntermNode *root,
                                                            TSymbolTable *symbolTable);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_HLSL_REWRITE_EXPRESSIONS_WITH_SHADER_STORAGE_BLOCK_H_
