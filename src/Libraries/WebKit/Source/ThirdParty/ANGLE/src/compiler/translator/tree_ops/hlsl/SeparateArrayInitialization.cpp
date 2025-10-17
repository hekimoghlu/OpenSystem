/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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

#include "compiler/translator/tree_ops/hlsl/SeparateArrayInitialization.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class SeparateArrayInitTraverser : private TIntermTraverser
{
  public:
    [[nodiscard]] static bool apply(TCompiler *compiler, TIntermNode *root);

  private:
    SeparateArrayInitTraverser();
    bool visitDeclaration(Visit, TIntermDeclaration *node) override;
};

bool SeparateArrayInitTraverser::apply(TCompiler *compiler, TIntermNode *root)
{
    SeparateArrayInitTraverser separateInit;
    root->traverse(&separateInit);
    return separateInit.updateTree(compiler, root);
}

SeparateArrayInitTraverser::SeparateArrayInitTraverser() : TIntermTraverser(true, false, false) {}

bool SeparateArrayInitTraverser::visitDeclaration(Visit, TIntermDeclaration *node)
{
    TIntermSequence *sequence = node->getSequence();
    TIntermBinary *initNode   = sequence->back()->getAsBinaryNode();
    if (initNode != nullptr && initNode->getOp() == EOpInitialize)
    {
        TIntermTyped *initializer = initNode->getRight();
        if (initializer->isArray() && !initializer->hasConstantValue())
        {
            // We rely on that array declarations have been isolated to single declarations.
            ASSERT(sequence->size() == 1);
            TIntermTyped *symbol      = initNode->getLeft();
            TIntermBlock *parentBlock = getParentNode()->getAsBlock();
            ASSERT(parentBlock != nullptr);

            TIntermSequence replacements;

            TIntermDeclaration *replacementDeclaration = new TIntermDeclaration();
            replacementDeclaration->appendDeclarator(symbol);
            replacementDeclaration->setLine(symbol->getLine());
            replacements.push_back(replacementDeclaration);

            TIntermBinary *replacementAssignment =
                new TIntermBinary(EOpAssign, symbol, initializer);
            replacementAssignment->setLine(symbol->getLine());
            replacements.push_back(replacementAssignment);

            mMultiReplacements.emplace_back(parentBlock, node, std::move(replacements));
        }
    }
    return false;
}

}  // namespace

bool SeparateArrayInitialization(TCompiler *compiler, TIntermNode *root)
{
    return SeparateArrayInitTraverser::apply(compiler, root);
}

}  // namespace sh
