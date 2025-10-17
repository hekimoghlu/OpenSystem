/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
// This mutating tree traversal works around a bug in the HLSL compiler optimizer with "pow" that
// manifests under the following conditions:
//
// - If pow() has a literal exponent value
// - ... and this value is integer or within 10e-6 of an integer
// - ... and it is in {-4, -3, -2, 2, 3, 4, 5, 6, 7, 8}
//
// The workaround is to replace the pow with a series of multiplies.
// See http://anglebug.com/40096900

#ifndef COMPILER_TRANSLATOR_TREEOPS_HLSL_EXPANDINTEGERPOWEXPRESSIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_HLSL_EXPANDINTEGERPOWEXPRESSIONS_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermNode;
class TSymbolTable;

[[nodiscard]] bool ExpandIntegerPowExpressions(TCompiler *compiler,
                                               TIntermNode *root,
                                               TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_HLSL_EXPANDINTEGERPOWEXPRESSIONS_H_
