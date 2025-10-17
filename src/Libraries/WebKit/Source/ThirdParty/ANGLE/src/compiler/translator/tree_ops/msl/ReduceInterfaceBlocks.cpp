/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include <algorithm>
#include <unordered_map>

#include "compiler/translator/IntermRebuild.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/msl/AstHelpers.h"
#include "compiler/translator/msl/TranslatorMSL.h"
#include "compiler/translator/tree_ops/SeparateDeclarations.h"
#include "compiler/translator/tree_ops/msl/ReduceInterfaceBlocks.h"

using namespace sh;

////////////////////////////////////////////////////////////////////////////////

namespace
{

class Reducer : public TIntermRebuild
{
    std::unordered_map<const TInterfaceBlock *, const TVariable *> mLiftedMap;
    std::unordered_map<const TVariable *, const TVariable *> mInstanceMap;
    IdGen &mIdGen;

  public:
    Reducer(TCompiler &compiler, IdGen &idGen)
        : TIntermRebuild(compiler, true, false), mIdGen(idGen)
    {}

    PreResult visitDeclarationPre(TIntermDeclaration &declNode) override
    {
        ASSERT(declNode.getChildCount() == 1);
        TIntermNode &node = *declNode.getChildNode(0);

        if (TIntermSymbol *symbolNode = node.getAsSymbolNode())
        {
            const TVariable &var        = symbolNode->variable();
            const TType &type           = var.getType();
            const SymbolType symbolType = var.symbolType();
            if (const TInterfaceBlock *interfaceBlock = type.getInterfaceBlock())
            {
                if (symbolType == SymbolType::Empty)
                {
                    // Create instance variable
                    auto &structure =
                        *new TStructure(&mSymbolTable, interfaceBlock->name(),
                                        &interfaceBlock->fields(), interfaceBlock->symbolType());
                    auto &structVar = CreateStructTypeVariable(mSymbolTable, structure);

                    auto &instanceVar = CreateInstanceVariable(
                        mSymbolTable, structure, mIdGen.createNewName(interfaceBlock->name()),
                        TQualifier::EvqBuffer, &type.getArraySizes());
                    mLiftedMap[interfaceBlock] = &instanceVar;

                    TIntermNode *replacements[] = {
                        new TIntermDeclaration{new TIntermSymbol(&structVar)},
                        new TIntermDeclaration{new TIntermSymbol(&instanceVar)}};
                    return PreResult::Multi(std::begin(replacements), std::end(replacements));
                }
                else
                {
                    ASSERT(type.getQualifier() == TQualifier::EvqUniform);

                    auto &structure =
                        *new TStructure(&mSymbolTable, interfaceBlock->name(),
                                        &interfaceBlock->fields(), interfaceBlock->symbolType());
                    auto &structVar = CreateStructTypeVariable(mSymbolTable, structure);
                    auto &instanceVar =
                        CreateInstanceVariable(mSymbolTable, structure, Name(var),
                                               TQualifier::EvqBuffer, &type.getArraySizes());

                    mInstanceMap[&var] = &instanceVar;

                    TIntermNode *replacements[] = {
                        new TIntermDeclaration{new TIntermSymbol(&structVar)},
                        new TIntermDeclaration{new TIntermSymbol(&instanceVar)}};
                    return PreResult::Multi(std::begin(replacements), std::end(replacements));
                }
            }
        }

        return {declNode, VisitBits::Both};
    }

    PreResult visitSymbolPre(TIntermSymbol &symbolNode) override
    {
        const TVariable &var = symbolNode.variable();
        {
            auto it = mInstanceMap.find(&var);
            if (it != mInstanceMap.end())
            {
                return *new TIntermSymbol(it->second);
            }
        }
        if (const TInterfaceBlock *ib = var.getType().getInterfaceBlock())
        {
            auto it = mLiftedMap.find(ib);
            if (it != mLiftedMap.end())
            {
                return AccessField(*(it->second), Name(var));
            }
        }
        return symbolNode;
    }
};

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

bool sh::ReduceInterfaceBlocks(TCompiler &compiler, TIntermBlock &root, IdGen &idGen)
{
    Reducer reducer(compiler, idGen);
    if (!reducer.rebuildRoot(root))
    {
        return false;
    }

    if (!SeparateDeclarations(compiler, root, false))
    {
        return false;
    }

    return true;
}
