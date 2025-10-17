/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
// The SeparateArrayInitialization function splits each array initialization into a declaration and
// an assignment.
// Example:
//     type[n] a = initializer;
// will effectively become
//     type[n] a;
//     a = initializer;
//
// Note that if the array is declared as const, the initialization may still be split, making the
// AST technically invalid. Because of that this transformation should only be used when subsequent
// stages don't care about const qualifiers. However, the initialization will not be split if the
// initializer can be written as a HLSL literal.

#ifndef COMPILER_TRANSLATOR_TREEOPS_HLSL_SEPARATEARRAYINITIALIZATION_H_
#define COMPILER_TRANSLATOR_TREEOPS_HLSL_SEPARATEARRAYINITIALIZATION_H_

#include "common/angleutils.h"

namespace sh
{
class TCompiler;
class TIntermNode;

[[nodiscard]] bool SeparateArrayInitialization(TCompiler *compiler, TIntermNode *root);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_HLSL_SEPARATEARRAYINITIALIZATION_H_
