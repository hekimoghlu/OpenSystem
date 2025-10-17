/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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

#include "compiler/translator/tree_ops/hlsl/AggregateAssignStructsInSSBOs.h"

#include "compiler/translator/StaticType.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermNode_util.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

class AggregateAssignStructsInSSBOsTraverser : public TIntermTraverser
{
  public:
    AggregateAssignStructsInSSBOsTraverser(TSymbolTable *symbolTable)
        : TIntermTraverser(true, false, false, symbolTable)
    {}

  protected:
    bool visitBinary(Visit visit, TIntermBinary *node) override
    {
        // Replace all assignments to structs in SSBOs with field-by-field asignments.
        // TODO(anglebug.com/42265832): this implementation only works for the simple case
        // (assignment statement), not more complex cases such as assignment-as-expression or
        // functions with side effects in the RHS.
        const TStructure *s;
        if (node->getOp() != EOpAssign)
        {
            return true;
        }
        else if (!IsInShaderStorageBlock(node->getLeft()))
        {
            return true;
        }
        else if (!(s = node->getLeft()->getType().getStruct()))
        {
            return true;
        }
        ASSERT(node->getRight()->getType().getStruct() == s);
        auto *block = new TIntermBlock();
        for (int i = 0; i < static_cast<int>(s->fields().size()); ++i)
        {
            auto *left   = new TIntermBinary(EOpIndexDirectStruct, node->getLeft()->deepCopy(),
                                             CreateIndexNode(i));
            auto *right  = new TIntermBinary(EOpIndexDirectStruct, node->getRight()->deepCopy(),
                                             CreateIndexNode(i));
            auto *assign = new TIntermBinary(TOperator::EOpAssign, left, right);
            block->appendStatement(assign);
        }

        queueReplacement(block, OriginalNode::IS_DROPPED);
        return false;
    }
};

}  // namespace

bool AggregateAssignStructsInSSBOs(TCompiler *compiler,
                                   TIntermBlock *root,
                                   TSymbolTable *symbolTable)
{
    AggregateAssignStructsInSSBOsTraverser traverser(symbolTable);
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
