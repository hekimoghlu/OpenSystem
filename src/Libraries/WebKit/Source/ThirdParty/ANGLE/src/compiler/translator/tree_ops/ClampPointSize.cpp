/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ClampPointSize.cpp: Limit the value that is written to gl_PointSize.
//

#include "compiler/translator/tree_ops/ClampPointSize.h"

#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/BuiltIn.h"
#include "compiler/translator/tree_util/FindSymbolNode.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/RunAtTheEndOfShader.h"

namespace sh
{

bool ClampPointSize(TCompiler *compiler,
                    TIntermBlock *root,
                    float minPointSize,
                    float maxPointSize,
                    TSymbolTable *symbolTable)
{
    // Only clamp gl_PointSize if it's used in the shader.
    const TIntermSymbol *glPointSize = FindSymbolNode(root, ImmutableString("gl_PointSize"));
    if (glPointSize == nullptr)
    {
        return true;
    }

    TIntermTyped *pointSizeNode = glPointSize->deepCopy();

    TConstantUnion *minPointSizeConstant = new TConstantUnion();
    TConstantUnion *maxPointSizeConstant = new TConstantUnion();
    minPointSizeConstant->setFConst(minPointSize);
    maxPointSizeConstant->setFConst(maxPointSize);
    TIntermConstantUnion *minPointSizeNode =
        new TIntermConstantUnion(minPointSizeConstant, TType(EbtFloat, EbpHigh, EvqConst));
    TIntermConstantUnion *maxPointSizeNode =
        new TIntermConstantUnion(maxPointSizeConstant, TType(EbtFloat, EbpHigh, EvqConst));

    // clamp(gl_PointSize, minPointSize, maxPointSize)
    TIntermSequence clampArguments;
    clampArguments.push_back(pointSizeNode->deepCopy());
    clampArguments.push_back(minPointSizeNode);
    clampArguments.push_back(maxPointSizeNode);
    TIntermTyped *clampedPointSize =
        CreateBuiltInFunctionCallNode("clamp", &clampArguments, *symbolTable, 100);

    // gl_PointSize = clamp(gl_PointSize, minPointSize, maxPointSize)
    TIntermBinary *assignPointSize = new TIntermBinary(EOpAssign, pointSizeNode, clampedPointSize);

    return RunAtTheEndOfShader(compiler, root, assignPointSize, symbolTable);
}

}  // namespace sh
