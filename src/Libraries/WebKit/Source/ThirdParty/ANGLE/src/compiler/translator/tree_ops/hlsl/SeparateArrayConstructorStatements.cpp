/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
// SeparateArrayConstructorStatements splits statements that are array constructors and drops all of
// their constant arguments. For example, a statement like:
//   int[2](0, i++);
// Will be changed to:
//   i++;

#include "compiler/translator/tree_ops/hlsl/SeparateArrayConstructorStatements.h"

#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

void SplitConstructorArgs(const TIntermSequence &originalArgs, TIntermSequence *argsOut)
{
    for (TIntermNode *arg : originalArgs)
    {
        TIntermTyped *argTyped = arg->getAsTyped();
        if (argTyped->hasSideEffects())
        {
            TIntermAggregate *argAggregate = argTyped->getAsAggregate();
            if (argTyped->isArray() && argAggregate && argAggregate->isConstructor())
            {
                SplitConstructorArgs(*argAggregate->getSequence(), argsOut);
            }
            else
            {
                argsOut->push_back(argTyped);
            }
        }
    }
}

class SeparateArrayConstructorStatementsTraverser : public TIntermTraverser
{
  public:
    SeparateArrayConstructorStatementsTraverser();

    bool visitAggregate(Visit visit, TIntermAggregate *node) override;
};

SeparateArrayConstructorStatementsTraverser::SeparateArrayConstructorStatementsTraverser()
    : TIntermTraverser(true, false, false)
{}

bool SeparateArrayConstructorStatementsTraverser::visitAggregate(Visit visit,
                                                                 TIntermAggregate *node)
{
    TIntermBlock *parentAsBlock = getParentNode()->getAsBlock();
    if (!parentAsBlock)
    {
        return false;
    }
    if (!node->isArray() || !node->isConstructor())
    {
        return false;
    }

    TIntermSequence constructorArgs;
    SplitConstructorArgs(*node->getSequence(), &constructorArgs);
    mMultiReplacements.emplace_back(parentAsBlock, node, std::move(constructorArgs));

    return false;
}

}  // namespace

bool SeparateArrayConstructorStatements(TCompiler *compiler, TIntermBlock *root)
{
    SeparateArrayConstructorStatementsTraverser traverser;
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
