/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 6, 2022.
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
// SeparateStructFromUniformDeclarations: Separate struct declarations from uniform declarations.
//

#include "compiler/translator/tree_ops/SeparateStructFromUniformDeclarations.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/tree_util/ReplaceVariable.h"

namespace sh
{
namespace
{
// This traverser translates embedded uniform structs into a specifier and declaration.
// This makes the declarations easier to move into uniform blocks.
class Traverser : public TIntermTraverser
{
  public:
    explicit Traverser(TSymbolTable *symbolTable)
        : TIntermTraverser(true, false, false, symbolTable)
    {}

    bool visitDeclaration(Visit visit, TIntermDeclaration *decl) override
    {
        ASSERT(visit == PreVisit);

        if (!mInGlobalScope)
        {
            return true;
        }

        const TIntermSequence &sequence = *(decl->getSequence());
        ASSERT(sequence.size() == 1);
        TIntermTyped *declarator = sequence.front()->getAsTyped();
        const TType &type        = declarator->getType();

        if (type.isStructSpecifier() && type.getQualifier() == EvqUniform)
        {
            doReplacement(decl, declarator, type);
            return false;
        }

        return true;
    }

    void visitSymbol(TIntermSymbol *symbol) override
    {
        const TVariable *variable = &symbol->variable();
        if (mVariableMap.count(variable) > 0)
        {
            queueAccessChainReplacement(mVariableMap[variable]->deepCopy());
        }
    }

  private:
    void doReplacement(TIntermDeclaration *decl, TIntermTyped *declarator, const TType &oldType)
    {
        const TStructure *structure = oldType.getStruct();
        if (structure->symbolType() == SymbolType::Empty)
        {
            // Handle nameless structs: uniform struct { ... } variable;
            structure = new TStructure(mSymbolTable, kEmptyImmutableString, &structure->fields(),
                                       SymbolType::AngleInternal);
        }
        TType *namedType = new TType(structure, true);
        namedType->setQualifier(EvqGlobal);

        TVariable *structVariable =
            new TVariable(mSymbolTable, kEmptyImmutableString, namedType, SymbolType::Empty);
        TIntermSymbol *structDeclarator       = new TIntermSymbol(structVariable);
        TIntermDeclaration *structDeclaration = new TIntermDeclaration;
        structDeclaration->appendDeclarator(structDeclarator);

        TIntermSequence newSequence;
        newSequence.push_back(structDeclaration);

        // Redeclare the uniform with the (potentially) new struct type
        TIntermSymbol *asSymbol = declarator->getAsSymbolNode();
        ASSERT(asSymbol && asSymbol->variable().symbolType() != SymbolType::Empty);

        TIntermDeclaration *namedDecl = new TIntermDeclaration;
        TType *uniformType            = new TType(structure, false);
        uniformType->setQualifier(EvqUniform);
        uniformType->makeArrays(oldType.getArraySizes());

        TVariable *newVar        = new TVariable(mSymbolTable, asSymbol->getName(), uniformType,
                                          asSymbol->variable().symbolType());
        TIntermSymbol *newSymbol = new TIntermSymbol(newVar);
        namedDecl->appendDeclarator(newSymbol);

        newSequence.push_back(namedDecl);

        mVariableMap[&asSymbol->variable()] = newSymbol;

        mMultiReplacements.emplace_back(getParentNode()->getAsBlock(), decl,
                                        std::move(newSequence));
    }

    VariableReplacementMap mVariableMap;
};
}  // anonymous namespace

bool SeparateStructFromUniformDeclarations(TCompiler *compiler,
                                           TIntermBlock *root,
                                           TSymbolTable *symbolTable)
{
    Traverser separateStructDecls(symbolTable);
    root->traverse(&separateStructDecls);
    return separateStructDecls.updateTree(compiler, root);
}
}  // namespace sh
