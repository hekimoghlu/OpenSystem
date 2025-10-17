/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
// ReplaceVariable.cpp: Replace all references to a specific variable in the AST with references to
// another variable.

#include "compiler/translator/tree_util/ReplaceVariable.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class ReplaceVariableTraverser : public TIntermTraverser
{
  public:
    ReplaceVariableTraverser(const TVariable *toBeReplaced, const TIntermTyped *replacement)
        : TIntermTraverser(true, false, false),
          mToBeReplaced(toBeReplaced),
          mReplacement(replacement)
    {}

    void visitSymbol(TIntermSymbol *node) override
    {
        if (&node->variable() == mToBeReplaced)
        {
            queueReplacement(mReplacement->deepCopy(), OriginalNode::IS_DROPPED);
        }
    }

  private:
    const TVariable *const mToBeReplaced;
    const TIntermTyped *const mReplacement;
};

class ReplaceVariablesTraverser : public TIntermTraverser
{
  public:
    ReplaceVariablesTraverser(const VariableReplacementMap &variableMap)
        : TIntermTraverser(true, false, false), mVariableMap(variableMap)
    {}

    void visitSymbol(TIntermSymbol *node) override
    {
        auto iter = mVariableMap.find(&node->variable());
        if (iter != mVariableMap.end())
        {
            queueReplacement(iter->second->deepCopy(), OriginalNode::IS_DROPPED);
        }
    }

  private:
    const VariableReplacementMap &mVariableMap;
};

class GetDeclaratorReplacementsTraverser : public TIntermTraverser
{
  public:
    GetDeclaratorReplacementsTraverser(TSymbolTable *symbolTable,
                                       VariableReplacementMap *variableMap)
        : TIntermTraverser(true, false, false, symbolTable), mVariableMap(variableMap)
    {}

    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override
    {
        const TIntermSequence &sequence = *(node->getSequence());

        for (TIntermNode *decl : sequence)
        {
            TIntermSymbol *asSymbol = decl->getAsSymbolNode();
            TIntermBinary *asBinary = decl->getAsBinaryNode();

            if (asBinary != nullptr)
            {
                ASSERT(asBinary->getOp() == EOpInitialize);
                asSymbol = asBinary->getLeft()->getAsSymbolNode();
            }

            ASSERT(asSymbol);
            const TVariable &variable = asSymbol->variable();

            ASSERT(mVariableMap->find(&variable) == mVariableMap->end());

            const TVariable *replacementVariable = new TVariable(
                mSymbolTable, variable.name(), &variable.getType(), variable.symbolType());

            (*mVariableMap)[&variable] = new TIntermSymbol(replacementVariable);
        }

        return false;
    }

  private:
    VariableReplacementMap *mVariableMap;
};

}  // anonymous namespace

// Replaces every occurrence of a variable with another variable.
[[nodiscard]] bool ReplaceVariable(TCompiler *compiler,
                                   TIntermBlock *root,
                                   const TVariable *toBeReplaced,
                                   const TVariable *replacement)
{
    ReplaceVariableTraverser traverser(toBeReplaced, new TIntermSymbol(replacement));
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

[[nodiscard]] bool ReplaceVariables(TCompiler *compiler,
                                    TIntermBlock *root,
                                    const VariableReplacementMap &variableMap)
{
    ReplaceVariablesTraverser traverser(variableMap);
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

void GetDeclaratorReplacements(TSymbolTable *symbolTable,
                               TIntermBlock *root,
                               VariableReplacementMap *variableMap)
{
    GetDeclaratorReplacementsTraverser traverser(symbolTable, variableMap);
    root->traverse(&traverser);
}

// Replaces every occurrence of a variable with a TIntermNode.
[[nodiscard]] bool ReplaceVariableWithTyped(TCompiler *compiler,
                                            TIntermBlock *root,
                                            const TVariable *toBeReplaced,
                                            const TIntermTyped *replacement)
{
    ReplaceVariableTraverser traverser(toBeReplaced, replacement);
    root->traverse(&traverser);
    return traverser.updateTree(compiler, root);
}

}  // namespace sh
