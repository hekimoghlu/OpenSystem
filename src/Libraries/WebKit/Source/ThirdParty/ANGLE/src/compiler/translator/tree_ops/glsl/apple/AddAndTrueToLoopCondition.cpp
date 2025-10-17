/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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

#include "compiler/translator/tree_ops/glsl/apple/AddAndTrueToLoopCondition.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

// An AST traverser that rewrites for and while loops by replacing "condition" with
// "condition && true" to work around condition bug on Intel Mac.
class AddAndTrueToLoopConditionTraverser : public TIntermTraverser
{
  public:
    AddAndTrueToLoopConditionTraverser() : TIntermTraverser(true, false, false) {}

    bool visitLoop(Visit, TIntermLoop *loop) override
    {
        // do-while loop doesn't have this bug.
        if (loop->getType() != ELoopFor && loop->getType() != ELoopWhile)
        {
            return true;
        }

        // For loop may not have a condition.
        if (loop->getCondition() == nullptr)
        {
            return true;
        }

        // Constant true.
        TIntermTyped *trueValue = CreateBoolNode(true);

        // CONDITION && true.
        TIntermBinary *andOp = new TIntermBinary(EOpLogicalAnd, loop->getCondition(), trueValue);
        loop->setCondition(andOp);

        return true;
    }
};

}  // anonymous namespace

bool AddAndTrueToLoopCondition(TCompiler *compiler, TIntermNode *root)
{
    AddAndTrueToLoopConditionTraverser traverser;
    root->traverse(&traverser);
    return compiler->validateAST(root);
}

}  // namespace sh
