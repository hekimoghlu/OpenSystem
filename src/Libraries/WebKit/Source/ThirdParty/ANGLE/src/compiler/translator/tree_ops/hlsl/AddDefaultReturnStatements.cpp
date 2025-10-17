/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
// AddDefaultReturnStatements.cpp: Add default return statements to functions that do not end in a
//                                 return.
//

#include "compiler/translator/tree_ops/hlsl/AddDefaultReturnStatements.h"

#include "compiler/translator/Compiler.h"
#include "compiler/translator/IntermNode.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

bool NeedsReturnStatement(TIntermFunctionDefinition *node, TType *returnType)
{
    *returnType = node->getFunctionPrototype()->getType();
    if (returnType->getBasicType() == EbtVoid)
    {
        return false;
    }

    TIntermBlock *bodyNode    = node->getBody();
    TIntermBranch *returnNode = bodyNode->getSequence()->back()->getAsBranchNode();
    if (returnNode != nullptr && returnNode->getFlowOp() == EOpReturn)
    {
        return false;
    }

    return true;
}

}  // anonymous namespace

bool AddDefaultReturnStatements(TCompiler *compiler, TIntermBlock *root)
{
    TType returnType;
    for (TIntermNode *node : *root->getSequence())
    {
        TIntermFunctionDefinition *definition = node->getAsFunctionDefinition();
        if (definition != nullptr && NeedsReturnStatement(definition, &returnType))
        {
            TIntermBranch *branch = new TIntermBranch(EOpReturn, CreateZeroNode(returnType));

            TIntermBlock *bodyNode = definition->getBody();
            bodyNode->getSequence()->push_back(branch);
        }
    }

    return compiler->validateAST(root);
}

}  // namespace sh
