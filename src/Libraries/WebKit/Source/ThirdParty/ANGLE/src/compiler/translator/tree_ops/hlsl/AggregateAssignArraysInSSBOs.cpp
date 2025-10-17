/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/tree_ops/hlsl/AggregateAssignArraysInSSBOs.h"

#include "compiler/translator/StaticType.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

class AggregateAssignArraysInSSBOsTraverser : public TIntermTraverser
{
  public:
    AggregateAssignArraysInSSBOsTraverser(TSymbolTable *symbolTable)
        : TIntermTraverser(true, false, false, symbolTable)
    {}

  protected:
    bool visitBinary(Visit visit, TIntermBinary *node) override
    {
        // Replace all aggregate assignments to arrays in SSBOs with element-by-element assignments.
        // TODO(anglebug.com/42265833): this implementation only works for the simple case
        // (assignment statement), not more complex cases such as assignment-as-expression or
        // functions with side effects in the RHS.

        if (node->getOp() != EOpAssign)
        {
            return true;
        }
        else if (!node->getLeft()->getType().isArray())
        {
            return true;
        }
        else if (!IsInShaderStorageBlock(node->getLeft()))
        {
            return true;
        }
        const TType *mediumpIndexType = StaticType::Get<EbtInt, EbpMedium, EvqTemporary, 1, 1>();
        auto *indexVariable           = CreateTempVariable(mSymbolTable, mediumpIndexType);
        auto *indexInit =
            CreateTempInitDeclarationNode(indexVariable, CreateZeroNode(indexVariable->getType()));
        auto *arraySizeNode   = CreateIndexNode(node->getOutermostArraySize());
        auto *indexSymbolNode = CreateTempSymbolNode(indexVariable);
        auto *cond = new TIntermBinary(EOpLessThan, indexSymbolNode->deepCopy(), arraySizeNode);
        auto *indexIncrement =
            new TIntermUnary(EOpPreIncrement, indexSymbolNode->deepCopy(), nullptr);
        auto *forLoopBody = new TIntermBlock();
        auto *indexedLeft =
            new TIntermBinary(EOpIndexDirect, node->getLeft(), indexSymbolNode->deepCopy());
        auto *indexedRight =
            new TIntermBinary(EOpIndexDirect, node->getRight(), indexSymbolNode->deepCopy());
        auto *assign = new TIntermBinary(TOperator::EOpAssign, indexedLeft, indexedRight);
        forLoopBody->appendStatement(assign);
        auto *forLoop = new TIntermLoop(ELoopFor, indexInit, cond, indexIncrement, forLoopBody);
        queueReplacement(forLoop, OriginalNode::IS_DROPPED);
        return false;
    }
};

}  // namespace

bool AggregateAssignArraysInSSBOs(TCompiler *compiler,
                                  TIntermBlock *root,
                                  TSymbolTable *symbolTable)
{
    AggregateAssignArraysInSSBOsTraverser traverser(symbolTable);
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
