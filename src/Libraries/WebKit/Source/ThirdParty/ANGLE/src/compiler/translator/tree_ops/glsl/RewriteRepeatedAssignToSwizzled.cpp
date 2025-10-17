/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
// RewriteRepeatedAssignToSwizzled.cpp: Rewrite expressions that assign an assignment to a swizzled
// vector, like:
//     v.x = z = expression;
// to:
//     z = expression;
//     v.x = z;
//
// Note that this doesn't handle some corner cases: expressions nested inside other expressions,
// inside loop headers, or inside if conditions.

#include "compiler/translator/tree_ops/glsl/RewriteRepeatedAssignToSwizzled.h"

#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class RewriteAssignToSwizzledTraverser : public TIntermTraverser
{
  public:
    [[nodiscard]] static bool rewrite(TCompiler *compiler, TIntermBlock *root);

  private:
    RewriteAssignToSwizzledTraverser();

    bool visitBinary(Visit, TIntermBinary *node) override;

    void nextIteration();

    bool didRewrite() { return mDidRewrite; }

    bool mDidRewrite;
};

// static
bool RewriteAssignToSwizzledTraverser::rewrite(TCompiler *compiler, TIntermBlock *root)
{
    RewriteAssignToSwizzledTraverser rewrite;
    do
    {
        rewrite.nextIteration();
        root->traverse(&rewrite);
        if (!rewrite.updateTree(compiler, root))
        {
            return false;
        }
    } while (rewrite.didRewrite());

    return true;
}

RewriteAssignToSwizzledTraverser::RewriteAssignToSwizzledTraverser()
    : TIntermTraverser(true, false, false), mDidRewrite(false)
{}

void RewriteAssignToSwizzledTraverser::nextIteration()
{
    mDidRewrite = false;
}

bool RewriteAssignToSwizzledTraverser::visitBinary(Visit, TIntermBinary *node)
{
    TIntermBinary *rightBinary = node->getRight()->getAsBinaryNode();
    TIntermBlock *parentBlock  = getParentNode()->getAsBlock();
    if (parentBlock && node->isAssignment() && node->getLeft()->getAsSwizzleNode() && rightBinary &&
        rightBinary->isAssignment())
    {
        TIntermSequence replacements;
        replacements.push_back(rightBinary);
        TIntermTyped *rightAssignmentTargetCopy = rightBinary->getLeft()->deepCopy();
        TIntermBinary *lastAssign =
            new TIntermBinary(node->getOp(), node->getLeft(), rightAssignmentTargetCopy);
        replacements.push_back(lastAssign);
        mMultiReplacements.emplace_back(parentBlock, node, std::move(replacements));
        mDidRewrite = true;
        return false;
    }
    return true;
}

}  // anonymous namespace

bool RewriteRepeatedAssignToSwizzled(TCompiler *compiler, TIntermBlock *root)
{
    return RewriteAssignToSwizzledTraverser::rewrite(compiler, root);
}

}  // namespace sh
