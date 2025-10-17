/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
// FoldExpressions.cpp: Fold expressions. This may fold expressions so that the qualifier of the
// folded node differs from the qualifier of the original expression, so it needs to be done after
// parsing and validation of qualifiers is complete. Expressions that are folded:
//  1. Ternary ops with a constant condition.
//  2. Sequence aka comma ops where the left side has no side effects.
//  3. Any expressions containing any of the above.

#include "compiler/translator/tree_ops/FoldExpressions.h"

#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class FoldExpressionsTraverser : public TIntermTraverser
{
  public:
    FoldExpressionsTraverser(TDiagnostics *diagnostics)
        : TIntermTraverser(true, false, false), mDiagnostics(diagnostics), mDidReplace(false)
    {}

    bool didReplace() { return mDidReplace; }

    void nextIteration() { mDidReplace = false; }

  protected:
    bool visitTernary(Visit visit, TIntermTernary *node) override
    {
        TIntermTyped *folded = node->fold(mDiagnostics);
        if (folded != node)
        {
            queueReplacement(folded, OriginalNode::IS_DROPPED);
            mDidReplace = true;
            return false;
        }
        return true;
    }

    bool visitAggregate(Visit visit, TIntermAggregate *node) override
    {
        TIntermTyped *folded = node->fold(mDiagnostics);
        if (folded != node)
        {
            queueReplacement(folded, OriginalNode::IS_DROPPED);
            mDidReplace = true;
            return false;
        }
        return true;
    }

    bool visitBinary(Visit visit, TIntermBinary *node) override
    {
        TIntermTyped *folded = node->fold(mDiagnostics);
        if (folded != node)
        {
            queueReplacement(folded, OriginalNode::IS_DROPPED);
            mDidReplace = true;
            return false;
        }
        return true;
    }

    bool visitUnary(Visit visit, TIntermUnary *node) override
    {
        TIntermTyped *folded = node->fold(mDiagnostics);
        if (folded != node)
        {
            queueReplacement(folded, OriginalNode::IS_DROPPED);
            mDidReplace = true;
            return false;
        }
        return true;
    }

    bool visitSwizzle(Visit visit, TIntermSwizzle *node) override
    {
        TIntermTyped *folded = node->fold(mDiagnostics);
        if (folded != node)
        {
            queueReplacement(folded, OriginalNode::IS_DROPPED);
            mDidReplace = true;
            return false;
        }
        return true;
    }

  private:
    TDiagnostics *mDiagnostics;
    bool mDidReplace;
};

}  // anonymous namespace

bool FoldExpressions(TCompiler *compiler, TIntermBlock *root, TDiagnostics *diagnostics)
{
    FoldExpressionsTraverser traverser(diagnostics);
    do
    {
        traverser.nextIteration();
        root->traverse(&traverser);
        if (!traverser.updateTree(compiler, root))
        {
            return false;
        }
    } while (traverser.didReplace());

    return true;
}

}  // namespace sh
